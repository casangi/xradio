import os
import shutil
import requests
import zipfile

from tqdm import tqdm

FILE_ID = {
    "AA2-Mid-sim_00000.ms": {
        "file": "AA2-Mid-sim_00000.ms.zip",
        "id": "2buf75ebhurjlfhe123qs",
        "rlkey": "8wzgiavzxfp4aza4sx84qanid&dl",
    },
    "Antennae_M8.img.zarr": {
        "file": "Antennae_M8.img.zarr.zip",
        "id": "9v0a1rv7nzm3kqte0u7vp",
        "rlkey": "ws6vd0jbo1dg7jvxsit42q5ba&dl",
    },
    "demo_simulated.im": {
        "file": "demo_simulated.im.zip",
        "id": "z87gibxshwg9e2h155ukk",
        "rlkey": "bn7uvs697wtedij63fa2hu7ed&dl",
    },
    "Antennae_North.cal.lsrk.ms": {
        "file": "Antennae_North.cal.lsrk.ms.zip",
        "id": "olx5qv9avdxxiyjlhlwx2",
        "rlkey": "trrqy43rfcqj4blf9robz4p47&dl",
    },
    "Antennae_North.cal.lsrk.vis.zarr": {
        "file": "Antennae_North.cal.lsrk.vis.zarr.zip",
        "id": "9hcunmq3iqtfiww593nrp",
        "rlkey": "7fingboduee7logszh25n95x5&dl",
    },
    "Antennae_North.cal.lsrk.split.ms": {
        "file": "Antennae_North.cal.lsrk.split.ms.zip",
        "id": "j2e5pd4y7ppvw9efxdfdj",
        "rlkey": "hlb85n40vtac3k9nna14giwsf&dl",
    },
    "Antennae_North.cal.lsrk.split.vis.zarr": {
        "file": "Antennae_North.cal.lsrk.split.ms.zip",
        "id": "hctg3tegkl6ttf2kol0wh",
        "rlkey": "7hxaag4vqk0d674v3368dudad&dl",
    },
    "no_mask.im": {
        "file": "no_mask.zip",
        "id": "4azivw1q7vby4ffawy0tt",
        "rlkey": "91g0hxwx6x4892aisbj5u195z&dl",
    },
}


def download(file, folder="."):
    if os.path.exists("/".join((folder, file))):
        print("File exists.")
        return

    if file not in FILE_ID.keys():
        print("Requested file not found")

        return

    fullname = FILE_ID[file]["file"]
    id = FILE_ID[file]["id"]
    rlkey = FILE_ID[file]["rlkey"]

    url = "https://www.dropbox.com/scl/fi/{id}/{file}?rlkey={rlkey}".format(
        id=id, file=fullname, rlkey=rlkey
    )
    print("url", url)
    headers = {"user-agent": "Wget/1.16 (linux-gnu)"}

    r = requests.get(url, stream=True, headers=headers)
    total = int(r.headers.get("content-length", 0))

    fullname = "/".join((folder, fullname))

    with open(fullname, "wb") as fd, tqdm(
        desc=fullname, total=total, unit="iB", unit_scale=True, unit_divisor=1024
    ) as bar:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                size = fd.write(chunk)
                bar.update(size)

    if zipfile.is_zipfile(fullname):
        shutil.unpack_archive(filename=fullname, extract_dir=folder)

        # Let's clean up after ourselves
        os.remove(fullname)
