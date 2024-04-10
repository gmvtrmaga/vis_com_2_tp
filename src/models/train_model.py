# -*- coding: utf-8 -*-
import logging
import sys
import os
import csv

from torch import manual_seed, save

from pathlib import Path

from train_utils import getTrainTestDataLoaders, trainModel

from strategies.ConvModel import ConvModelTrainConfig
from strategies.SqueezeNetModel import SqueezeNetModelTrainConfig
from strategies.ResNet18Model import ResNet18ModelTrainConfig

import click
from dotenv import find_dotenv, load_dotenv

DEFAULT_BATCH_SIZE = 64
DEFAULT_TORCH_SEED = 42
DEFAULT_N_FREEZE = 1

TRAIN_HISTORY_FILENAME = 'history.csv'
TRAINED_MODEL_FILENAME = 'model.mdl'


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path())
@click.argument('log_output_filepath', type=click.Path())
@click.argument('image_size', type=click.INT)
@click.argument('model_to_train')
@click.argument('train_epochs', type=click.INT)
@click.option('--random_state', default=DEFAULT_TORCH_SEED, type=click.INT)
@click.option('--n_freeze', default=DEFAULT_N_FREEZE, type=click.INT)
@click.option('--batch_size', default=DEFAULT_BATCH_SIZE, type=click.INT)
def main(input_filepath, model_filepath, log_output_filepath,
         image_size, model_to_train, train_epochs, random_state=DEFAULT_TORCH_SEED,
         n_freeze=DEFAULT_N_FREEZE,batch_size=DEFAULT_BATCH_SIZE):
    """ Trains the selected CNN model using the seleected dataset 
    located at input filepath. 
    """
    manual_seed(random_state)

    logger = logging.getLogger(__name__)
    logger.info('Preparing augmentation and datasets')

    train_loader, valid_loader = \
        getTrainTestDataLoaders(input_filepath, image_size, batch_size)

    match model_to_train:
        case "ConvModel":
            logger.info('ConvModel SELECTED')
            train_config = ConvModelTrainConfig(image_size)
        case "SqueezeNet":
            logger.info('SqueezeNet SELECTED')
            train_config = SqueezeNetModelTrainConfig(n_freeze)
        case "ResNet18":
            logger.info('ResNet18 SELECTED')
            train_config = ResNet18ModelTrainConfig(n_freeze)                      
        case _:
            logger.error("Invalid option")
            sys.exit()

    logger.info('Training started')
    train_history, trained_model = \
        trainModel(train_config.model, train_config.optimizer, train_config.loss,
                   train_config.metric, train_loader, valid_loader, train_epochs,
                   tensorboard_log=True,  register_path=log_output_filepath, image_size=image_size)

    model_path = os.path.join(model_filepath, TRAINED_MODEL_FILENAME)
    save(trained_model.state_dict(), model_path)

    history_path = os.path.join(log_output_filepath, TRAIN_HISTORY_FILENAME)
    with open(history_path, "w", newline='\n') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(train_history.keys())
        writer.writerows(zip(*train_history.values()))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
