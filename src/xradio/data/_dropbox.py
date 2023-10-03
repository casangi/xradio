import os
import shutil
import requests
import zipfile

from tqdm import tqdm
#
#https://www.dropbox.com/scl/fi/x8tp0wu21gssbd1gxrnmy/Antennae_North.cal.lsrk.vis.zarr.zip?rlkey=l9jdr6tvyq4pe3381gukuly0d&dl=0

FILE_ID = {
  'demo_simulated.im':
  {
    'file':'demo_simulated.im.zip',
    'id':'z87gibxshwg9e2h155ukk',
    'rlkey':'bn7uvs697wtedij63fa2hu7ed&dl'
  },
 'Antennae_North.cal.lsrk.vis.zarr':
  {
    'file':'Antennae_North.cal.lsrk.vis.zarr.zip',
    'id':'x8tp0wu21gssbd1gxrnmy',
    'rlkey':'l9jdr6tvyq4pe3381gukuly0d&dl'
  },
  
  'Antennae_North.cal.lsrk.ms':
   {
    'file':'Antennae_North.cal.lsrk.ms.zip',
    'id':'olx5qv9avdxxiyjlhlwx2',
    'rlkey':'trrqy43rfcqj4blf9robz4p47&dl'
   },
}

def download(file, folder='.'):
  if os.path.exists('/'.join((folder, file))):
    print("File exists.")
    return
    
  if file not in FILE_ID.keys():
    print("Requested file not found")
    
    return 
  
  fullname=FILE_ID[file]['file']
  id=FILE_ID[file]['id']
  rlkey=FILE_ID[file]['rlkey']
    
  url = 'https://www.dropbox.com/scl/fi/{id}/{file}?rlkey={rlkey}'.format(id=id, file=fullname, rlkey=rlkey)
    
  headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}

  r = requests.get(url, stream=True, headers=headers)
  total = int(r.headers.get('content-length', 0))

  fullname = '/'.join((folder, fullname))

  with open(fullname, 'wb') as fd, tqdm(
    desc=fullname,
    total=total,
    unit='iB',
    unit_scale=True,
    unit_divisor=1024) as bar:
      for chunk in r.iter_content(chunk_size=1024):
        if chunk:
          size=fd.write(chunk)
          bar.update(size)
                
  if zipfile.is_zipfile(fullname):                
    shutil.unpack_archive(filename=fullname, extract_dir=folder)
    
    # Let's clean up after ourselves
    os.remove(fullname)
