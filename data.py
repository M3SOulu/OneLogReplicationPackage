import csv
import itertools
import multiprocessing
import os
import pickle
import re
import shutil
import warnings
from functools import cached_property
from pathlib import Path
from typing import List, Sequence

import fire
import inflection
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, ConcatDataset, DataLoader

from encoder import CharacterEncoder
from utils import sliding_window, dataset_summary, dataset_random_split, NoisifyingDatasetWrapper, \
    AheadOfTimeDatasetWrapper, pad2d, TargetInverterDatasetWrapper, Table


class OneLogDataset(LightningDataModule):
    def __init__(self, batch_size: int = 128,
                 train_dataset_names: List[str] = None,
                 test_dataset_names: List[str] = None,
                 train_validation_proportions=(0.8, 0.2),
                 num_workers=None,
                 min_shape=(0, 0),
                 noise_ratio=0.0,
                 aot_ratio=0.0):
        super().__init__()
        self.train_dataset_names = BaseDataset.all_dataset_names() if train_dataset_names is None else train_dataset_names
        self.test_dataset_names = BaseDataset.all_dataset_names() if test_dataset_names is None else test_dataset_names
        self.train_validation_proportions = train_validation_proportions
        self.noise_ratio = noise_ratio
        self.aot_ratio = aot_ratio
        self.encoder = CharacterEncoder()
        self.num_workers = multiprocessing.cpu_count() if num_workers is None else num_workers
        self.batch_size = batch_size
        self.dataset_train = None
        self.dataset_val = None
        self.datasets_test = None
        self.min_shape = min_shape

    def setup(self, stage=None):
        if self.dataset_train is not None and self.dataset_val is not None and self.datasets_test is not None:
            return
        test_sets = []
        train_sets = []
        for name, dataset in BaseDataset.all_datasets().items():
            if not any(name in ds for ds in set(self.train_dataset_names + self.test_dataset_names)):
                continue
            ds = dataset(self.encoder)
            setattr(self, f"{dataset.dataset_name()}_dataset", ds)
            train_split, test_split = ds.train_test_split()

            if name in self.train_dataset_names:
                train_sets.append(train_split)
            if f"I{name}" in self.train_dataset_names:
                inverted_train_split = TargetInverterDatasetWrapper(train_split)
                inverted_train_split.name = f"inverted_{name}"
                train_sets.append(inverted_train_split)

            if name in self.test_dataset_names:
                test_split.name = name
                test_sets.append(test_split)
            if f"N{name}" in self.test_dataset_names:
                assert 0 < self.noise_ratio < 1, "A noisy dataset requested for testing, yet the noise ratio value is invalid"
                noisy_test_split = NoisifyingDatasetWrapper(test_split, self.noise_ratio)
                noisy_test_split.name = f"noisy_{name}"
                test_sets.append(noisy_test_split)
            if f"A{name}" in self.test_dataset_names:
                assert 0 < self.aot_ratio < 1, "Ahead of time dataset is requested for testing, yet the AoT ratio value is invalid"
                noisy_test_split = AheadOfTimeDatasetWrapper(test_split, self.aot_ratio)
                noisy_test_split.name = f"aot_{name}"
                test_sets.append(noisy_test_split)
            if f"I{name}" in self.test_dataset_names:
                noisy_test_split = TargetInverterDatasetWrapper(test_split)
                noisy_test_split.name = f"inverted_{name}"
                test_sets.append(noisy_test_split)

        self.datasets_test = test_sets
        self.dataset_train, self.dataset_val = dataset_random_split(
            ConcatDataset(train_sets), self.train_validation_proportions)

        dataset_summary([
                            ('train', self.dataset_train),
                            ('validation', self.dataset_val),
                        ] + [(f'test_{self.test_dataset_names[i]}', tds) for i, tds in enumerate(test_sets)])

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.dataset_train, self.batch_size, shuffle=True, num_workers=self.num_workers,
                          collate_fn=self.collate)

    def val_dataloader(self, *args, **kwargs) -> Sequence[DataLoader]:
        return [DataLoader(dataset, self.batch_size, shuffle=False, num_workers=self.num_workers,
                           collate_fn=self.collate) for dataset in [self.dataset_val] + self.datasets_test]

    # def test_dataloaders(self) -> List[Tuple[str, DataLoader]]:
    #     return [(name, DataLoader(dataset, self.batch_size, shuffle=False, num_workers=self.num_workers,
    #                               collate_fn=self.collate)) for name, dataset in self.datasets_test]

    def collate(self, data):
        x, y = list(zip(*data))
        lens = [len(e) for e in x]
        x = pad2d(x, self.min_shape)
        y = torch.tensor(y, dtype=torch.float)
        return x, lens, y

    @property
    def encoder_characters(self):
        return self.encoder.dictionary_length

    @property
    def testset_count(self):
        return len(self.test_dataset_names)


class BaseDataset(Dataset):
    __CACHE_DONE_FILE = ".done"
    __DATA_OBJECT_CACHE_FILE_NAME = "data_object.pkl"
    __PARSER_FILE_PERSISTENCE_HANDLER = "parsed_templates.drain3"
    __DATASET_DIR = os.environ['DATASET_DIR'] if 'DATASET_DIR' in os.environ else str(Path.home() / 'Datasets')

    def __init__(self, directory, encoder, use_cache=True, validate_data=True, train_test_split_proportions=(0.5, 0.5),
                 data_type="both"):
        assert type(self).__name__[-7:] == BaseDataset.__name__[-7:], "Dataset name convention adaptation error"
        assert data_type in ["both", "anomaly", "normal"]
        self.directory = os.path.join(self.__DATASET_DIR, directory)
        self.encoder = encoder
        self.train_test_split_proportions = train_test_split_proportions
        self.data_type = data_type
        if self.cache_available and use_cache:
            self.__data = self.cache_load()
        else:
            self.__data = self.load_data()
            if validate_data:
                self.validate_data()
            if use_cache:
                self.cache_save(BaseDataset.__DATA_OBJECT_CACHE_FILE_NAME, self.__data)
                self.set_cache_available()

    def train_test_split(self):
        return dataset_random_split(self, self.train_test_split_proportions)

    @classmethod
    def dataset_name(cls):
        return inflection.underscore(cls.__name__[:-7])

    @staticmethod
    def all_datasets():
        return {cls.dataset_name(): cls for cls in BaseDataset.__subclasses__()}

    @staticmethod
    def all_dataset_names():
        return list(BaseDataset.all_datasets().keys()) + \
               [f"N{name}" for name in BaseDataset.all_datasets()] + \
               [f"A{name}" for name in BaseDataset.all_datasets()] + \
               [f"I{name}" for name in BaseDataset.all_datasets()]

    @cached_property
    def data(self):
        if self.data_type == "both":
            return [(sequence, label) for label, sequences in enumerate(self.__data) for sequence in sequences]
        elif self.data_type == "anomaly":
            return self.__data[1]
        elif self.data_type == "normal":
            return self.__data[0]
        else:
            raise RuntimeError("Invalid data class")

    @property
    def data_normal(self):
        return self.__data[0]

    @property
    def data_anomaly(self):
        return self.__data[1]

    @property
    def cache_available(self):
        return os.path.exists(self.cache_file(BaseDataset.__CACHE_DONE_FILE))

    def set_cache_available(self):
        with self.cache_open(BaseDataset.__CACHE_DONE_FILE, "w") as file:
            file.write("Cache complete")

    @property
    def cache_dir(self):
        return f".cache_{type(self).__name__}"

    @property
    def parsed_dataset(self):
        template_miner = self.load_parser()
        parsed_dataset = []
        for seq, target in self.data:
            x = np.array([template_miner.match_log_message(mess)["cluster_id"] for mess in seq])
            y = target
            parsed_dataset.append((x, y))
        return parsed_dataset

    @property
    def parsed_text_dataset(self):
        template_miner = self.load_parser()
        parsed_text_dataset = []
        for seq, target in self.data:
            x = [template_miner.match_log_message(mess)["template_mined"] for mess in seq]
            y = target
            parsed_text_dataset.append((x, y))
        return parsed_text_dataset

    def cache_file(self, filename):
        return os.path.join(self.cache_dir, filename)

    def cache_open(self, filename, mode="r"):
        if not os.path.isdir(self.cache_dir):
            os.mkdir(self.cache_dir)
        return open(self.cache_file(filename), mode)

    def cache_save(self, filename, obj):
        with self.cache_open(filename, "wb") as file:
            pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

    def cache_load(self):
        with self.cache_open(BaseDataset.__DATA_OBJECT_CACHE_FILE_NAME, "rb") as file:
            return pickle.load(file)

    def load_parser(self):
        template_miner = TemplateMiner(FilePersistence(self.cache_file(BaseDataset.__PARSER_FILE_PERSISTENCE_HANDLER)))
        if len(template_miner.drain.clusters) == 0:
            for seq, _ in self.data:
                for mess in seq:
                    template_miner.add_log_message(mess)
        return template_miner

    def clear_cache(self):
        shutil.rmtree(self.cache_dir)

    def open(self, filename, mode="r", encoding=None, errors=None):
        return open(os.path.join(self.directory, filename), mode, encoding=encoding, errors=errors)

    def load_data(self):
        raise NotImplementedError()

    def validate_data(self):
        assert self.__data
        if len(self.__data[0]) < len(self.__data[1]):
            warnings.warn("Normal samples are less than anomaly samples")
        assert self.data
        for seq, lb in self.data:
            assert isinstance(seq, list)
            assert isinstance(lb, int) and (lb == 1 or lb == 0)
            for line in seq:
                assert isinstance(line, str)
                assert len(line) > 0

    def __getitem__(self, item):
        if self.data_type == "both":
            s, y = self.data[item]
            x = self.encoder.encode(s)
            y = torch.tensor(y)
            return x, y
        elif self.data_type in ["anomaly", "normal"]:
            return self.encoder.encode(self.data[item])
        else:
            raise RuntimeError("Invalid data class")

    def __len__(self):
        return len(self.data)


class HDFSDataset(BaseDataset):

    def __init__(self, encoder, *args, **kwargs):
        super().__init__(directory="hdfs", encoder=encoder, train_test_split_proportions=(0.8, 0.2), *args, **kwargs)

    def load_data(self):
        data_table = [[], []]
        block_label = self.load_table_block_label()
        block_sequence = self.load_table_block_sequence()
        for block, sequence in block_sequence.items():
            data_table[block_label[block]].append(sequence)
        return data_table

    def load_table_block_label(self):
        with self.open("anomaly_label.csv") as file:
            csv_reader = csv.reader(file)
            return {blk_id: 0 if label == "Normal" else 1 for blk_id, label in csv_reader}

    def load_table_block_sequence(self):
        regex_blk_id = re.compile("blk_-?[0-9]+")
        table = {}
        with self.open("HDFS.log") as file:
            for line in file:
                message = line.strip().split(maxsplit=5)[-1]
                blk_id = regex_blk_id.search(message).group()
                if blk_id not in table:
                    table[blk_id] = []
                message = message.replace(blk_id, "blk_")
                table[blk_id].append(message)
        return table


class OpenstackDataset(BaseDataset):

    def __init__(self, encoder, *args, **kwargs):
        super().__init__("openstack", encoder, train_test_split_proportions=(0.9, 0.1), *args, **kwargs)

    def load_data(self):
        normal1 = self.load_table_instance_sequence("openstack_normal1.log")
        normal2 = self.load_table_instance_sequence("openstack_normal2.log")
        abnormal = self.load_table_instance_sequence("openstack_abnormal.log")
        return [[sequence for _, sequence in itertools.chain(normal1.items(), normal2.items())],
                [sequence for _, sequence in abnormal.items()]]

    def load_table_instance_sequence(self, filename):
        regex_log_format = re.compile(r"\[([a-zA-Z0-9- ]*)\](?: \[instance: ((?:[a-zA-Z0-9]*-?)*)])? (.*)")
        table = {}
        with self.open(filename) as file:
            for line in file:
                last_partition = line.split(maxsplit=6)[-1]
                match_obj = regex_log_format.findall(last_partition)
                if match_obj:
                    _, instance_id, message = match_obj[0]
                    if not instance_id:
                        instance_id = "*"
                else:
                    instance_id, message = "-", last_partition
                if instance_id in "-*":
                    continue
                if instance_id not in table:
                    table[instance_id] = []
                table[instance_id].append(message)
        return table


class HadoopDataset(BaseDataset):

    def __init__(self, encoder, *args, **kwargs):
        self.window_stride = 8
        self.max_sequence_len = 64
        super().__init__("hadoop", encoder, train_test_split_proportions=(0.9, 0.1), *args, **kwargs)

    def load_data(self):
        data_table = [[], []]
        anomaly_labels = self.load_anomaly_labels()
        for sequence, label in self.load_sequence_label():
            data_table[0 if label not in anomaly_labels else 1].append(sequence)
        return data_table

    def load_anomaly_labels(self):
        with self.open("abnormal_label.txt") as file:
            return set(label.strip() for label in file)

    def load_sequence_label(self):
        sequence_label = []
        for directory in os.listdir(self.directory):
            log_dir = os.path.join(self.directory, directory)
            if not os.path.isdir(log_dir):
                continue
            for file in os.listdir(log_dir):
                filename = os.path.join(directory, file)
                sequence = self.load_sequence(filename)
                for seq in sliding_window(sequence, self.max_sequence_len, self.window_stride):
                    sequence_label.append((seq, directory))
        return sequence_label

    def load_sequence(self, filename):
        regex_log_format = re.compile(r"\d*-\d*-\d* \d*:\d*:\d*,\d* [a-zA-Z]* \[.*] [^\s]*: (.*)")
        sequence = []
        with self.open(filename) as file:
            for line in file:
                match_obj = regex_log_format.match(line)
                if match_obj:
                    sequence.append(match_obj.groups()[0].strip())
        return sequence


class BGLDataset(BaseDataset):
    def __init__(self, encoder):
        super().__init__("bgl", encoder, train_test_split_proportions=(0.9, 0.1))

    def load_data(self):
        data_table = [set(), set()]
        for message, label in self.log_lines():
            data_table[label].add(message)
        return [
            [[s] for s in data_table[0]],
            [[s] for s in data_table[1]]
        ]

    def log_lines(self):
        log_regex = re.compile(r"(FATAL|INFO|WARNING|DEBUG|SEVERE|ERROR|FAILURE) ?(.*)")
        with self.open("BGL.log", encoding='UTF-8') as file:
            for line in file:
                label = 0 if line[0] == "-" else 1
                message = log_regex.findall(line)[0][1]
                if len(message) == 0:
                    continue
                yield message, label


class ThunderbirdDataset(BaseDataset):

    def __init__(self, encoder):
        super().__init__("thunderbird", encoder, train_test_split_proportions=(0.9, 0.1))

    def load_data(self):
        data_table = [set(), set()]
        with self.open("Thunderbird.log", encoding='utf-8', errors='ignore') as file:
            for line in file:
                label = 0 if line[0] == "-" else 1
                message = line.strip().split(maxsplit=9 - label)[-1]
                if len(message) == 0:
                    raise ValueError("Empty message")
                data_table[label].add(message)
        return [
            [[s] for s in data_table[0]],
            [[s] for s in data_table[1]]
        ]


class SpiritDataset(BaseDataset):
    def __init__(self, encoder):
        super().__init__("spirit", encoder, train_test_split_proportions=(0.9, 0.1))

    def load_data(self):
        data_table = [set(), set()]
        with self.open("spirit2.log", encoding='utf-8', errors='ignore') as file:
            for line in file:
                splits = line.strip().split(maxsplit=8)
                message = splits[-1]
                label = 0 if splits[0] == '-' else 1
                if len(message) == 0:
                    raise ValueError("Empty message")
                data_table[label].add(message)
        return [
            [[s] for s in data_table[0]],
            [[s] for s in data_table[1]]
        ]


class LibertyDataset(BaseDataset):
    def __init__(self, encoder):
        super().__init__("liberty", encoder, train_test_split_proportions=(0.9, 0.1))

    def load_data(self):
        data_table = [set(), set()]
        with self.open("liberty2.log", encoding='utf-8', errors='ignore') as file:
            for line in file:
                splits = line.strip().split(maxsplit=8)
                message = splits[-1]
                label = 0 if splits[0] == '-' else 1
                if len(message) == 0:
                    raise ValueError("Empty message")
                data_table[label].add(message)
        return [
            [[s] for s in data_table[0]],
            [[s] for s in data_table[1]]
        ]


class DataCommands:
    def recap(self, *datasets: str, auto_title=True):
        name_dataset_table = BaseDataset.all_datasets()
        if datasets == ('all',):
            datasets = [(name, name_dataset_table[name](None)) for name in name_dataset_table.keys()]
        dataset_summary(datasets, auto_title)


if __name__ == '__main__':
    fire.Fire(DataCommands)
