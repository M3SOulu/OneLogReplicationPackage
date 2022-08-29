import copy
import csv
import json
import random

import math
import numpy as np
import tabulate
import torch
from torch.utils.data import random_split, Dataset

from models import BaseModel


def pad2d(x, min_shape=(0, 0), dtype=torch.long):
    max_d1, max_d2 = min_shape
    for sx1 in x:
        max_d1 = max(len(sx1), max_d1)
        for sx2 in sx1:
            max_d2 = max(len(sx2), max_d2)
    padded_x = torch.zeros(len(x), max_d1, max_d2, dtype=dtype)
    for i, sx1 in enumerate(x):
        for j, sx2 in enumerate(sx1):
            for k, value in enumerate(sx2):
                padded_x[i, j, k] = value
    return padded_x


def sliding_window(sequence, windows_size, stride=1):
    seq_len = len(sequence)
    slides = seq_len - windows_size
    if slides < 1:
        yield sequence
    for i in range(0, slides, stride):
        p = min(i, seq_len - windows_size)
        yield sequence[p: p + windows_size]


class Table:
    def __init__(self, headers, title_headers=True):
        self.title_headers = title_headers
        assert type(headers) in [int, list]
        if isinstance(headers, int):
            self.table = []
            for _ in range(headers):
                self.table.append([])
        elif isinstance(headers, list):
            self.table = {}
            for header in headers:
                self.table[header] = []

    def add(self, *args, **kwargs):
        if isinstance(self.table, dict):
            for field in self.table:
                self.table[field].append(kwargs[field])
        elif isinstance(self.table, list):
            for i, field in enumerate(args):
                self.table[i].append(field)

    def print(self):
        args = {
            "tabular_data": {k.replace('_', ' ').title(): v for k, v in self.table.items()},
            "tablefmt": "grid",
            "numalign": "right"
        }
        if isinstance(self.table, dict):
            args["headers"] = "keys"
        print(tabulate.tabulate(**args))


class StatisticsCalculator:
    def __init__(self):
        self.__sum = 0
        self.__n = 0
        self.__max = float("-inf")
        self.__min = float("inf")

    def add(self, x):
        self.__n += 1
        self.__sum += x

        if self.__max < x:
            self.__max = x

        if self.__min > x:
            self.__min = x

    @property
    def mean(self):
        return self.__sum / self.__n

    @property
    def count(self):
        return self.__n

    @property
    def max(self):
        return self.__max

    @property
    def min(self):
        return self.__min


def dataset_summary(datasets, auto_title=True):
    summary = Table(['name', 'samples', 'anomaly', 'normal', 'ratio'], title_headers=auto_title)
    for name, ds in datasets:
        # ds = name_dataset_table[name](None)
        anomalies = len(ds.data_anomaly) if hasattr(ds, 'data_anomaly') else sum(l.item() for _, l in ds)
        summary.add(
            name=name,
            samples=len(ds),
            anomaly=anomalies,
            normal=len(ds) - anomalies,
            ratio=anomalies / (len(ds) - anomalies)
        )
    if len(datasets) > 1:
        anomaly = sum(summary.table['anomaly'])
        normal = sum(summary.table['normal'])
        summary.add(
            name='Total' if auto_title else 'total',
            samples=sum(summary.table['samples']),
            anomaly=anomaly,
            normal=normal,
            ratio=anomaly / normal
        )
    summary.print()


def export_numpy(dataset, x_export_file, y_export_file, invert_targets=False):
    x = []
    y = []
    for seq, target in dataset.parsed_dataset:
        x.append(np.array(seq))
        if invert_targets:
            y.append(1 - target)
        else:
            y.append(target)

    x = np.array(x, dtype=object)
    y = np.array(y, dtype=object)
    np.save(x_export_file, x)
    np.save(y_export_file, y)


def sample_agreement_proportion(dataset1, dataset2):
    eqs = 0
    for (x1, _), (x2, _) in zip(dataset1, dataset2):
        if json.dumps(x1) == json.dumps(x2):
            eqs += 1
    return eqs / len(dataset1)


def dataset_random_split(dataset, proportions):
    assert sum(proportions) == 1, "Invalid proportions"
    l = len(dataset)
    len_train = math.floor(l * proportions[0])
    len_val = l - len_train
    return random_split(dataset, [len_train, len_val])


def export_characters_tensorflow_projector_visualization(model: BaseModel, character_encoder,
                                                         vector_path, metadata_path):
    with open(vector_path, "w") as vector_file, open(metadata_path, "w") as metadata_file:
        vector_tsv = csv.writer(vector_file, delimiter="\t")
        metadata_tsv = csv.writer(metadata_file, delimiter="\t")
        for character, vector in zip(character_encoder.all_keys(), model.character_embedding_vectors()):
            vector_tsv.writerow(vector)
            metadata_tsv.writerow([character])


class NoisifyingDatasetWrapper(Dataset):
    def __init__(self, dataset, noise_ratio, duplicate_proportion=0.3, removal_proportion=0.3, shuffle_proportion=0.3):
        self.dataset = dataset
        self.duplicate_proportion = duplicate_proportion
        self.remove_proportion = removal_proportion
        self.shuffle_proportion = shuffle_proportion
        no_change_ratio = 1 - noise_ratio
        assert no_change_ratio > 0, "Invalid noise ratio"
        self.noise_type_list = random.choices(["no", "duplicate", "remove", "shuffle"],
                                              [no_change_ratio, noise_ratio / 3, noise_ratio / 3, noise_ratio / 3],
                                              k=len(dataset))
        self.item_cache = {}

    def __getitem__(self, item):
        modification_type = self.noise_type_list[item]
        if modification_type == "no":
            return self.dataset[item]

        if item in self.item_cache:
            return self.item_cache[item]

        elif modification_type == "duplicate":
            x, y = copy.deepcopy(self.dataset[item])
            for _ in range(int(self.duplicate_proportion * len(x))):
                index = random.randrange(0, len(x))
                event = x[index]
                x.insert(index, event)
            self.item_cache[item] = x, y
            return x, y

        elif modification_type == "remove":
            if len(self.dataset[item][0]) <= self.remove_proportion ** -1:
                return self.dataset[item]
            x, y = copy.deepcopy(self.dataset[item])
            for _ in range(int(self.remove_proportion * len(x))):
                index = random.randrange(0, len(x))
                del x[index]
            self.item_cache[item] = x, y
            return x, y

        elif modification_type == "shuffle":
            if len(self.dataset[item][0]) <= self.shuffle_proportion ** -1:
                return self.dataset[item]
            x, y = copy.deepcopy(self.dataset[item])
            shuffle_len = int(self.shuffle_proportion * len(x))
            index = random.randrange(0, len(x) - shuffle_len)
            sub_x = x[index: index + shuffle_len]
            random.shuffle(sub_x)
            x[index: index + shuffle_len] = sub_x
            self.item_cache[item] = x, y
            return x, y
        else:
            raise RuntimeError(f"Invalid modification type for item {item}")

    def __len__(self):
        return len(self.dataset)


class AheadOfTimeDatasetWrapper(Dataset):
    def __init__(self, dataset, aot_ratio):
        assert 0 < aot_ratio < 1, "Invalid noise ratio"
        self.dataset = dataset
        self.aot_ratio = aot_ratio

    def __getitem__(self, item):
        x, y = self.dataset[item]
        ei = round((1 - self.aot_ratio) * len(x))
        return x[:ei], y

    def __len__(self):
        return len(self.dataset)


class TargetInverterDatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, item):
        x, y = self.dataset[item]
        return x, 1 - y

    def __len__(self):
        return len(self.dataset)
