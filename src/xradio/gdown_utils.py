import os
import re
import gdown
import shutil
import json

from prettytable import PrettyTable

gdown_ids = {
    "Antennae_South.cal.ms": "1f6a6hge0mDZVi3wUJYjRiY3Cyr9HvqZz",
    "Antennae_North.cal.ms": "1sASTyp4gr4PzWZwJr_ZHEdkqcYjF86BT",
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


def gdown_data(ms_name, download_folder="."):
    assert (
        ms_name in gdown_ids
    ), "Measurement set not available. Available measurement sets are:" + str(
        gdown_ids.keys()
    )

    id = gdown_ids[ms_name]
    create_folder(download_folder)
    check_download(ms_name, download_folder, id)


def list_datasets():
    table = PrettyTable()
    table.field_names = ["Measurement Table"]  #  ,"Description"]
    table.align = "l"

    for key, _ in gdown_ids.items():
        #        basename = key.split('.')[0]
        #        file = ''.join((basename, '.json'))
        #        path = os.path.dirname(__file__)
        table.add_row([str(key)])

    print(table)
