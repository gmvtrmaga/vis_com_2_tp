import zipfile


def unzip_file(input_filepath, output_dirpath):
    with zipfile.ZipFile(input_filepath, "r") as zip_ref:
        zip_ref.extractall(output_dirpath)
