import os
import xradio

def download(file, folder='.', unpack=False, source='dropbox'):

  if os.path.exists(folder) == False:
    os.mkdir(folder)
  
  if source == 'gdrive':
    from xradio.data._google_drive import download
    download(file=file, folder=folder, unpack=unpack)

  elif source == 'dropbox':
    from xradio.data._dropbox import download
    download(file=file, folder=folder)

  else:
    print("unknown source or issue found")
    pass
