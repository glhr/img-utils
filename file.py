import pathlib
import os.path

def get_current_directory():
    return str(pathlib.Path(__file__).parent.absolute())


def get_working_directory():
    return str(pathlib.Path().absolute())


def get_color_from_filename(filename):
    color = filename.split(".")[0].split("\\")[-1].split("/")[-1].split("-")[0]
    return color


def get_filename_from_path(path, extension=True):
    if extension:
        return path.split("\\")[-1].split("/")[-1]
    return path.split("\\")[-1].split("/")[-1].split(".")[-2]


def file_exists(path):
    return os.path.isfile(path)
