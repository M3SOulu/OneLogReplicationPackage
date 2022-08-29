from pytorch_lightning.cli import LightningCLI, LightningArgumentParser
from models import HierarchicalCnnModel
from data import OneLogDataset


class OneLogTrainerCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments('data.encoder_characters', 'model.max_embedding_indices', apply_on='instantiate')
        parser.link_arguments('data.testset_count', 'model.testset_count', apply_on='instantiate')


def main():
    OneLogTrainerCLI(model_class=HierarchicalCnnModel, datamodule_class=OneLogDataset, save_config_callback=False)


if __name__ == '__main__':
    main()
