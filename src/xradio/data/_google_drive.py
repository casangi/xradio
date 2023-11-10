import os
import re
import gdown
import shutil
import json

from astrohack._utils._tools import _remove_suffix

from prettytable import PrettyTable

FILE_ID = {
    "Antennae_South.cal.ms": "1f6a6hge0mDZVi3wUJYjRiY3Cyr9HvqZz",
    "Antennae_North.cal.ms": "1sASTyp4gr4PzWZwJr_ZHEdkqcYjF86BT",
    "demo_simulated.im": "1esOGbRMMEZXvTxQ_bdcw3PaZLg6XzHC5",
}


def check_download(name, folder, id):
    fullname = os.path.join(folder, name)
    if not os.path.exists(fullname):
        url = "https://drive.google.com/u/0/uc?id=" + id + "&export=download"
        gdown.download(url, fullname + ".zip")
        shutil.unpack_archive(filename=fullname + ".zip", extract_dir=folder)


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def build_folder_structure(dataname, resultname):
    create_folder(dataname)
    create_folder(resultname)


def download(file, folder=".", unpack=False):
    """Allows access to stakeholder and unit testing data and configuration files via gdown.

    :param file: File to download for gdirve storage. A list of the available measurement sets can be accessed via `astrohack.datasets.list_datasets()`.
    :type file: str
    :param folder: Destination folder if not the current directory, defaults to '.'
    :type folder: str, optional
    :param unpack: Unzip file, defaults to False
    :type unpack: bool, optional
    """

    # print("The google-drive download option is deprecated and will be removed soon!! Please use the dropbox option in download.")

    if file == "vla-test":
        matched = [
            (key, value)
            for key, value in FILE_ID.items()
            if re.search(
                r"^vla.+(before|after).split.+(holog|image|panel|point).*zarr$", key
            )
        ]
        files = files = list(dict(matched).keys())

    elif file == "alma-test":
        matched = [
            (key, value)
            for key, value in FILE_ID.items()
            if re.search(r"^alma.split.+(holog|image|panel|point).*zarr$", key)
        ]
        files = list(dict(matched).keys())

    else:
        files = [file]

    for file in files:
        assert (
            file in FILE_ID
        ), "File {file} not available. Available files are:".format(file=file) + str(
            FILE_ID.keys()
        )

        id = FILE_ID[file]
        create_folder(folder)

        fullname = os.path.join(folder, file)

        if os.path.exists(fullname) or os.path.exists(fullname + ".zip"):
            continue

        if unpack:
            fullname = fullname + ".zip"

        url = "https://drive.google.com/u/0/uc?id=" + id + "&export=download"
        gdown.download(url, fullname)

        # Unpack results
        if unpack:
            shutil.unpack_archive(filename=fullname, extract_dir=folder)

            # Let's clean up after ourselves
            os.remove(fullname)


def list_datasets():
    table = PrettyTable()
    table.field_names = ["Measurement Table", "Description"]
    table.align = "l"

    for key, _ in FILE_ID.items():
        basename = key.split(".")[0]
        file = "".join((basename, ".json"))
        path = os.path.dirname(__file__)

        with open(
            "{path}/data/.file_meta_data/{file}".format(path=path, file=file)
        ) as file:
            ms_info = json.load(file)

        description_string = f"""
        Observer: {ms_info['observer']}
        Project:{ms_info['project']}
        Elapsed Time: {ms_info['elapsed time']}
        Observed: {ms_info['observed']}
        SPW-ID: {ms_info['spwID']}
        Name: {ms_info['name']}
        Channels: {ms_info['channels']}
        Frame: {ms_info['frame']}
        Channel0: {ms_info['chan0']} MHz
        Channel Width: {ms_info['chan-width']} kHz
        Total Bandwidth: {ms_info['total-bandwidth']} kHz
        Center Frequency: {ms_info['center-frequency']} MHz
        Correlations: {ms_info['corrs']}
        RA: {ms_info['ra']}
        DEC: {ms_info['dec']}
        EPOCH: {ms_info['epoch']}
        Notes: {ms_info['notes']}
        """

        table.add_row([str(key), description_string])

    print(table)


def _remove_suffix(input_string, suffix):
    """
    Removes extension suffixes from file names
    Args:
        input_string: filename string
        suffix: The suffix to be removed

    Returns: the input string minus suffix

    """
    if suffix and input_string.endswith(suffix):
        return input_string[: -len(suffix)]

    return input_string
