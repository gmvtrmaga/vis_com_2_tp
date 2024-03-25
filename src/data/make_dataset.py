# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv

from file_utils import unzip_file, file_setup, clean_directory


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_dirpath", type=click.Path())
def main(input_filepath, output_dirpath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    logger.info("Cleaning directory")
    clean_directory(output_dirpath)

    logger.info("Extracting data from zip file")
    unzip_file(input_filepath, output_dirpath)

    logger.info("Renaming files")
    file_setup(output_dirpath)

    logger.info("Completed")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
