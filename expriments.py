import argparse
import copy
import os
import random
import shutil
import sys
import fire
from abc import ABC, abstractmethod
from pathlib import Path

import inflection
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F

from models import HierarchicalCnnModel
from utils import pad2d


class OldExperiments:
    def __init__(self, model: str, out: str, device=None):
        model = Path(model)
        out = Path(out)
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = HierarchicalCnnModel.load_from_checkpoint(model).to(self.device)
        self.model.freeze()
        if out.exists():
            assert out.is_dir(), f"Output directory exists at {out}, yet it is not a directory. Consider removing it and running the experiment again."
            shutil.rmtree(out)
        out.mkdir()
        self.output_dir = out
        self.cache = {}

    def rca(self):
        from data import HDFSDataset
        from encoder import CharacterEncoder
        encoder = CharacterEncoder()
        dataset = HDFSDataset(encoder, data_type="anomaly")
        sns.set(rc={'figure.figsize': (30, 30)})
        for idx, sample in enumerate(dataset):
            if not 20 < len(sample) < 40 or idx < 1000:
                continue
            for removal_ratio in [0.5] * 30:
                m_r = self.generate_removal_matrix(sample, samples_per_anomaly=256, removal_ratio=removal_ratio)
                m_n = self.generate_sample_from_removal_matrix(sample, m_r)
                x, lens = self.collate(m_n)
                x = x.to(self.model.device)
                y_h = self.model(x, lens)
                # fig, axes = plt.subplots(3, 1)
                # axes[0].hist(y_h.cpu().numpy(), 10, facecolor="blue")
                # plt.subplots(2, 1, 1)
                # plt.hist(y_h.cpu().numpy(), 25, facecolor="blue")
                avg_prob = torch.mean(y_h).cpu().numpy()
                # y_h = torch.logit(y_h)
                # y_h = y_h / (1 - y_h)
                # axes[1].hist(y_h.cpu().numpy(), 10, facecolor="red")
                # y_h = torch.log(y_h)
                # axes[2].hist(y_h.cpu().numpy(), 10, facecolor="green")
                # fig.show()
                # plt.subplots(2, 1, 2)
                # plt.hist(y_h.cpu().numpy(), 25, facecolor="red")
                # plt.show()
                y_p = np.expand_dims(y_h.cpu().numpy(), -1)
                suspiciousness = (m_r * y_p).mean(0)
                sus_events = np.argsort(suspiciousness)[:3]
                print(f"{idx:5} - {removal_ratio} - {avg_prob:.2f}: {sus_events}")

    @staticmethod
    def generate_removal_matrix(x, samples_per_anomaly=128, removal_ratio=0.5):
        removal_matrix = np.random.random((samples_per_anomaly, len(x)))
        removal_matrix = removal_matrix < removal_ratio
        return removal_matrix

    @staticmethod
    def generate_sample_from_removal_matrix(x, removal_matrix):
        new_x = []
        for removal_vector in removal_matrix:
            sx = copy.deepcopy(x)
            for index in reversed(np.where(removal_vector)[0]):
                del sx[index]
            new_x.append(sx)
        return new_x

    @staticmethod
    def collate(x):
        lens = [len(e) for e in x]
        x = pad2d(x, min_shape=(6, 0))
        return x, lens

    def integrated_gradients(self):
        from data import HDFSDataset
        from encoder import CharacterEncoder
        from captum.attr import LayerIntegratedGradients
        encoder = CharacterEncoder()
        normal_dataset = HDFSDataset(encoder, data_type="normal")
        anomaly_dataset = HDFSDataset(encoder, data_type="anomaly")

        integrated_gradients = LayerIntegratedGradients(self.model, self.model.character_embedding)
        for n_idx, normal_sample in enumerate(random.choices(normal_dataset, k=100)):
            if len(normal_sample) < 7:
                print(n_idx, "-", "Passed!")
                continue
            if self.model(pad2d([normal_sample]).to(self.model.device), None).item() > 0.1:
                print(n_idx, "-", "Passed!, too high score")
                continue
            for a_idx, anomaly_sample in enumerate(random.choices(anomaly_dataset, k=100)):
                if len(anomaly_sample) < 7:
                    print(n_idx, a_idx, "Passed!")
                    continue
                x = pad2d([normal_sample, anomaly_sample]).to(self.model.device)
                pred = self.model(x, None)
                if pred[1] - pred[0] < 0.7:
                    print(n_idx, a_idx, "Difference is not big enough")
                    continue
                baseline = x[:1]
                target_input = x[1:]
                # lens = [len(e) for e in x]

                attributions = integrated_gradients.attribute(target_input, baseline, n_steps=25,
                                                              additional_forward_args=(None,))
                heatmap, _ = attributions[0].max(dim=-1)
                abs_max = heatmap.max().abs().item()

                # heatmap -= heatmap.mean()
                # if dataset.data_type == "normal":
                #     heatmap *= -1
                # heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
                # heatmap *= -1
                # heatmap /= torch.norm(heatmap)

                heatmap = heatmap.sum(dim=1, keepdims=True)
                heatmap = heatmap.cpu().detach().numpy()
                new_hmap = np.zeros_like(heatmap)
                annotations = np.full(heatmap.shape, ' ', dtype=object)
                for i, item in enumerate(anomaly_sample):
                    annotations[i, 0] = encoder.decode(item)
                    new_hmap[i, 0] = heatmap[i, 0]

                # for i, row in enumerate(anomaly_sample):
                #     for j, item in enumerate(row):
                #         if item in (0, 95):
                #             annotations[i, j] = ' '
                #             new_hmap[i, j] = 0
                #         else:
                #             annotations[i, j] = encoder[item]
                #             new_hmap[i, j] = heatmap[i, j]

                fig, ax = plt.subplots(figsize=(50, 50))
                sns.heatmap(new_hmap, annot=annotations, linewidths=0, fmt='', annot_kws={"fontsize": 24}, square=False,
                            ax=ax, cmap="coolwarm", vmin=-abs_max, vmax=abs_max)
                plt.savefig(f'out/{n_idx}_{a_idx}.png')
                print("Heapmap for:", n_idx, a_idx)
                # fig.show()
                plt.close(fig)
                pass

    def message_embedding(self):
        from encoder import CharacterEncoder
        encoder = CharacterEncoder()
        cache = {}

        def embed(message):
            if message in cache:
                return cache[message]
            x = torch.tensor(encoder.encode([message]), dtype=torch.long).permute(1, 0)
            x = x.to(self.model.device)
            x = self.model.embed_message(x)[0]
            cache[message] = x
            return x

        def similarity(a, b):
            return F.cosine_similarity(a, b, dim=0).item()

        df = pd.read_csv('hdfs.csv')
        all_matched = 0
        for idx1, row1 in df.iterrows():
            emb1 = embed(row1['message'])
            best_match = max(
                ((row2, similarity(emb1, embed(row2['message']))) for idx2, row2 in df.iterrows() if idx1 != idx2),
                key=lambda x: x[1])
            if row1['template'] == best_match[0]['template']:
                print(idx1, "Matched")
                all_matched += 1
            else:
                print(idx1, "Not Matched", best_match[1])
                print(row1['message'])
                print(best_match[0]['message'])
            print("Total matched:", all_matched)
            print('-' * 100)

    def embed(self, message):
        if message in self.cache:
            return self.cache[message]
        x = torch.tensor(self.encoder.encode([message]), dtype=torch.long)
        x = x.to(self.model.device)
        if x.shape[1] < 16:
            x = torch.nn.functional.pad(x, (0, 16 - x.shape[0], 0, 0))
        x = self.model.embed_message(x)[0]
        x = x.cpu().detach().numpy()
        self.cache[message] = x
        return x


class Experiments:
    def __init__(self, model: str, device=None):
        model = Path(model)
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = HierarchicalCnnModel.load_from_checkpoint(model).to(self.device)
        self.model.freeze()
        self.cache = {}

    def _embed(self, message):
        if message in self.cache:
            return self.cache[message]
        x = torch.tensor(self.encoder.encode([message]), dtype=torch.long)
        x = x.to(self.model.device)
        if x.shape[1] < 16:
            x = torch.nn.functional.pad(x, (0, 16 - x.shape[0], 0, 0))
        x = self.model.embed_message(x)[0]
        x = x.cpu().detach().numpy()
        self.cache[message] = x
        return x

    @staticmethod
    def _validate_output_directory(output_dir):
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        assert output_dir.is_dir(), f"Output directory exists at {output_dir}, yet it is not a directory. Consider removing it and running the experiment again."
        return output_dir

    def silhouette(self, message_template_csv_path: str, output_directory_path: str = None):
        """
        Experiment parsing
        :param output_directory_path:
        :param message_template_csv_path: the input csv file that contains messages and templates.
        It should have two columns: 'message' and 'template'.
        :return:
        """
        from encoder import CharacterEncoder
        from sklearn.metrics import silhouette_score
        self.encoder = CharacterEncoder()

        df = pd.read_csv(message_template_csv_path)
        if output_directory_path is None:
            output_directory_path = Path(message_template_csv_path).parent / 'out'
        else:
            output_directory_path = self._validate_output_directory(output_directory_path)
        vectors_path = output_directory_path / 'vectors.tsv'
        output_path = output_directory_path / 'meta.tsv'
        vectors = []
        for idx1, row1 in df.iterrows():
            emb = self._embed(row1['message'])
            vectors.append(emb)
        vectors = np.array(vectors)
        print(f"{silhouette_score(vectors, df['template'])=}")
        np.savetxt(vectors_path, vectors, delimiter='\t')
        df.to_csv(output_path, index=False, sep='\t')
        print(f"vectors exported to {vectors_path}")
        print(f"meta data exported {output_path}")
        print("head on to https://projector.tensorflow.org/ to visualize them")

    def integrated_gradients(self):
        from data import HDFSDataset
        from encoder import CharacterEncoder
        from captum.attr import LayerIntegratedGradients
        encoder = CharacterEncoder()
        normal_dataset = HDFSDataset(encoder, data_type="normal")
        anomaly_dataset = HDFSDataset(encoder, data_type="anomaly")

        integrated_gradients = LayerIntegratedGradients(self.model, self.model.character_embedding)
        for n_idx, normal_sample in enumerate(random.choices(normal_dataset, k=100)):
            if len(normal_sample) < 7:
                print(n_idx, "-", "Passed!")
                continue
            if self.model(pad2d([normal_sample]).to(self.model.device)).item() > 0.1:
                print(n_idx, "-", "Passed!, too high score")
                continue
            for a_idx, anomaly_sample in enumerate(random.choices(anomaly_dataset, k=100)):
                if len(anomaly_sample) < 7:
                    print(n_idx, a_idx, "Passed!")
                    continue
                x = pad2d([normal_sample, anomaly_sample]).to(self.model.device)
                pred = self.model(x)
                if pred[1] - pred[0] < 0.7:
                    print(n_idx, a_idx, "Difference is not big enough")
                    continue
                baseline = x[:1]
                target_input = x[1:]
                # lens = [len(e) for e in x]

                attributions = integrated_gradients.attribute(target_input, baseline, n_steps=25)
                heatmap, _ = attributions[0].max(dim=-1)
                abs_max = heatmap.max().abs().item()

                # heatmap -= heatmap.mean()
                # if dataset.data_type == "normal":
                #     heatmap *= -1
                # heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
                # heatmap *= -1
                # heatmap /= torch.norm(heatmap)

                heatmap = heatmap.sum(dim=1, keepdims=True)
                heatmap = heatmap.cpu().detach().numpy()
                new_hmap = np.zeros_like(heatmap)
                annotations = np.full(heatmap.shape, ' ', dtype=object)
                for i, item in enumerate(anomaly_sample):
                    annotations[i, 0] = encoder.decode(item)
                    new_hmap[i, 0] = heatmap[i, 0]

                # for i, row in enumerate(anomaly_sample):
                #     for j, item in enumerate(row):
                #         if item in (0, 95):
                #             annotations[i, j] = ' '
                #             new_hmap[i, j] = 0
                #         else:
                #             annotations[i, j] = encoder[item]
                #             new_hmap[i, j] = heatmap[i, j]

                fig, ax = plt.subplots(figsize=(50, 50))
                sns.heatmap(new_hmap, annot=annotations, linewidths=0, fmt='', annot_kws={"fontsize": 24}, square=False,
                            ax=ax, cmap="rocket_r", vmin=0.02, vmax=abs_max)
                plt.savefig(f'experiment/out/{n_idx}_{a_idx}.png')
                print("Heapmap for:", n_idx, a_idx)
                # fig.show()
                plt.close(fig)
                pass


def umap_plot(vectors, templates, nn=15):
    from umap import UMAP
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import OrdinalEncoder
    ordinal_encoder = OrdinalEncoder()
    y = ordinal_encoder.fit_transform(np.expand_dims(np.array(templates), -1)).reshape((-1,))
    df = pd.DataFrame(UMAP(n_neighbors=nn).fit_transform(vectors, y=y))
    df['template'] = templates
    sns.relplot(data=df,
                x=0,
                y=1,
                palette=sns.color_palette("husl", len(ordinal_encoder.categories_[0])),
                hue='template',
                legend=False)
    plt.show()


def main():
    fire.Fire(Experiments)


def parse_cla():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", "-e", type=str, nargs=1, choices=[])
    parser.add_argument("--log-directory", type=str, default="lightning_logs")
    parser.add_argument("--version", "-v", type=int, default=-1)
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


def get_log_dir(args):
    version = args.version
    if version == -1:
        version = max([int(file_name[8:]) for file_name in os.listdir(args.log_directory)])
    return os.path.join(args.log_directory, f"version_{version}")


def load_model(log_dir):
    model_file_name = list(os.listdir(os.path.join(log_dir, "checkpoints")))[0]
    return HierarchicalCnnModel.load_from_checkpoint(os.path.join(log_dir, "checkpoints", model_file_name))


if __name__ == '__main__':
    fire.Fire(Experiments)
    # main()
