import zipfile
from os import walk, path, rename, listdir, remove
import shutil
from logging import Logger

GIT_KEEP = ".gitkeep"
NEGATIVE_DIRECTORY = "Normal"
POSITIVE_DIRECTORY = "DDH"


def clean_directory(target_directory):
    # Check if the target directory exists
    if path.exists(target_directory):
        # Iterate over all elements in the directory
        for element in listdir(target_directory):
            # Construct the full path to the element
            element_path = path.join(target_directory, element)
            # Check if it's a file
            if path.isfile(element_path) and element != GIT_KEEP:
                # Delete the file
                remove(element_path)
            # Check if it's a directory
            elif path.isdir(element_path):
                # Delete the directory and its contents recursively
                shutil.rmtree(element_path)


def unzip_file(input_filepath, output_dirpath):
    with zipfile.ZipFile(input_filepath, "r") as zip_ref:
        zip_ref.extractall(output_dirpath)


def file_setup(target_directory):

    TEMP_FILENAME_MODIFICATOR = "_temp"

    for root, dirs, files in walk(target_directory):
        for index, file_name in enumerate(files, start=0):

            if file_name != GIT_KEEP:
                old_file_path = path.join(root, file_name)
                new_file_name = (
                    str(index) + path.splitext(file_name)[1] + TEMP_FILENAME_MODIFICATOR
                )
                new_file_path = path.join(root, new_file_name)
                rename(old_file_path, new_file_path)

    for root, dirs, files in walk(target_directory):
        for index, file_name in enumerate(files, start=1):
            if file_name != GIT_KEEP:
                old_file_path = path.join(root, file_name)
                new_file_name = (
                    str(index)
                    + path.splitext(file_name[: -len(TEMP_FILENAME_MODIFICATOR)])[1]
                )
                new_file_path = path.join(root, new_file_name)
                rename(old_file_path, new_file_path)


def get_file_tree(target_directory):
    subdirectories_and_files = {}

    for root, dirs, files in walk(target_directory):
        # Relative path from the main directory
        relative_path = path.relpath(root, target_directory)

        # List of files in the current directory
        file_list = [file for file in files]

        if len(file_list) > 0 and GIT_KEEP not in file_list:
            # Store the relative path and list of files in the dictionary
            subdirectories_and_files[relative_path] = file_list

    return subdirectories_and_files


def check_file_tree(file_tree, logger: Logger):
    positive_count = 0
    negative_count = 0
    paired_subdirectories = {}

    # Check if there are pairs of subdirectories
    for directory, _files in file_tree.items():
        parent_directory = path.dirname(directory)

        if parent_directory in paired_subdirectories:
            if paired_subdirectories[parent_directory] is True:
                logger.warning(
                    "There are more than 2 directories in parent directory with name "
                    + parent_directory
                )
                return None
            else:
                paired_subdirectories[parent_directory] = True
        else:
            paired_subdirectories[parent_directory] = False

    for parent_directory, paired in paired_subdirectories.items():
        if not paired:
            logger.warning(
                "There is only one directory in parent directory with name "
                + parent_directory
            )
            return None

    # Iterate over the keys (directories) and values (lists of files) in the dictionary
    for directory, files in file_tree.items():
        nFiles = len(files)

        if POSITIVE_DIRECTORY in directory:
            if positive_count != nFiles:
                if positive_count == 0:
                    positive_count = nFiles
                else:
                    logger.warning(
                        "Directories do not have the same number of positive entries: "
                        + directory
                    )

                    return None
        elif NEGATIVE_DIRECTORY:
            if negative_count != nFiles:
                if negative_count == 0:
                    negative_count = nFiles
                else:
                    logger.warning(
                        "Directories do not have the same number of negative entries: "
                        + directory
                    )

                    return None
        else:
            logger.warning(
                "The following directory does not have the correct name: " + directory
            )
            return None

    return [positive_count, negative_count]
