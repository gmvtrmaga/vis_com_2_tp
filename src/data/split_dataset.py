# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv
from file_utils import (
    check_file_tree,
    clean_directory,
    get_file_tree,
    split_train_test_files,
)


@click.command()
@click.argument("input_dirpath", type=click.Path(exists=True))
@click.argument("output_dirpath", type=click.Path())
@click.option("--random_state", default=None, help="Seed for random number generation")
@click.option("--train_size", default=0.8, type=float, help="Proportion of ")
def main(input_dirpath, output_dirpath, random_state, train_size=0.8):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    fileTree = get_file_tree(input_dirpath)

    logger.info("Checking files...")
    file_count = check_file_tree(fileTree, logger)

    if file_count is not None:
        positive_count, negative_count = file_count
        logger.info(
            "Files successfully checked: "
            + str(positive_count)
            + " positives and "
            + str(negative_count)
            + " negatives. "
        )

        logger.info("Cleaning directory")
        clean_directory(output_dirpath)

        logger.info(
            "Shuffling data in train/test directories. Random state: "
            + str(random_state)
            + ". Train proportion: "
            + str(train_size)
        )
        split_train_test_files(
            input_dirpath,
            output_dirpath,
            fileTree,
            file_count,
            random_state,
            train_size,
        )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
