# -*- coding: utf-8 -*-
import logging
import sys
import click

from dotenv import find_dotenv, load_dotenv
from pathlib import Path

from PIL import Image

import torch
from train_utils import getDataTransoforms

from strategies.ConvModel import ConvModel
from strategies.ResNet18Model import CustomResNet18Net
from strategies.SqueezeNetModel import CustomSqueezeNet

@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("model_filepath", type=click.Path(exists=True))
@click.argument("model_to_train")
def main(input_filepath, model_filepath, model_to_train):
    """Makes a prediction for input_filepath using the previously persisted model (loaded from ../model_filepath)."""

    logger = logging.getLogger(__name__)
    logger.info("Making prediction")

    logger.info("Loading and transforming image")
    input_image = None 
    with open(input_filepath, "rb") as f:
        input_image = Image.open(f).convert("RGB")
    
    image_size = input_image.size[0]
    
    _, data_transforms = getDataTransoforms(image_size)

    model_input = data_transforms(input_image)   

    match model_to_train:
        case "ConvModel":
            logger.info("ConvModel SELECTED")
            model = ConvModel(image_size)
        case "SqueezeNet":
            logger.info("SqueezeNet SELECTED")
            model = CustomSqueezeNet()
        case "ResNet18":
            logger.info("ResNet18 SELECTED")
            model = CustomResNet18Net()
        case _:
            logger.error("Invalid option")
            sys.exit()

    logger.info("Loading model")
    model.load_state_dict(torch.load(model_filepath, map_location=torch.device('cpu')))
    model.eval()

    logger.info("Prediction result")
    prediction = model(model_input[None, :, :, :]) 
    if prediction <= 0.5:
        logger.info("Result: DDH detected")
    else:
        logger.info("Result: Normal")        

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()