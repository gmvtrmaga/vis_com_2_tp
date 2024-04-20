# -*- coding: utf-8 -*-
import logging
import sys
import click

from PIL import Image

from torch import load
from src.models.train_utils import getDataTransoforms

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
    with open(input_filepath, "r") as f:
        input_image = Image.open(f).convert("RGB")
    
    image_size = input_image.shape[0]
    
    _, data_transforms = getDataTransoforms(image_size)

    model_input = data_transforms(input_image)   

    match model_to_train:
        case "ConvModel":
            logger.info("ConvModel SELECTED")
            model = ConvModel(image_size)
        case "SqueezeNet":
            logger.info("SqueezeNet SELECTED")
            model = CustomResNet18Net()
        case "ResNet18":
            logger.info("ResNet18 SELECTED")
            model = CustomSqueezeNet()
        case _:
            logger.error("Invalid option")
            sys.exit()

    logger.info("Loading model")
    model.load_state_dict(load(model_filepath))

    logger.info("Prediction result")
    if model(model_input) > 0.5:
        logger.info("DDH detected")
    else:
        logger.info("Normal")
