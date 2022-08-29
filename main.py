import argparse
import os
import sys

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything

from callbacks import OverallTestDataset
from data import OneLogDataset, BaseDataset
from models import HierarchicalCnnModel


def main():
    args = parse_cla()
    print(args)
    dataset = OneLogDataset(args.batch_size,
                            train_dataset_names=args.train_datasets,
                            test_dataset_names=args.test_datasets,
                            noise_ratio=args.noise_ratio,
                            aot_ratio=args.aot_ratio
                            )
    # print("Training on", len(dataset.dataset_train), "sequence,", "validating on", len(dataset.dataset_val),
    #       "sequences")
    if args.multi_gpu and not __debug__:
        assert args.random_seed, "Random seed must be set if training on multiple GPUs"
        seed_everything(args.random_seed)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12123'
        gpus = -1
    else:
        gpus = 1
    trainer = Trainer(gpus=gpus,
                      callbacks=[
                          OverallTestDataset(),
                          ModelCheckpoint(filename="{epoch}-{val_f1}", monitor="val_f1", mode="max")
                      ],
                      log_every_n_steps=1)
    model = HierarchicalCnnModel(dataset.encoder.dictionary_length, args.model_dimension, lr=args.learning_rate)
    trainer.fit(model, datamodule=dataset)


def parse_cla():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-datasets", "-d", nargs="+", choices=BaseDataset.all_dataset_names(),
                        default=BaseDataset.all_dataset_names())
    parser.add_argument("--test-datasets", "-t", nargs="+", choices=BaseDataset.all_dataset_names(),
                        default=BaseDataset.all_dataset_names())
    parser.add_argument("--batch-size", "-b", type=int, default=128)
    parser.add_argument("--model-dimension", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--multi-gpu", action="store_true", default=False)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--noise-ratio", type=float, default=0)
    parser.add_argument("--aot-ratio", type=float, default=0)
    args = parser.parse_args(sys.argv[1:])
    return args


if __name__ == '__main__':
    main()
