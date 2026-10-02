"""
Configuration flags of the BDF loaders.

use_load_one_spw_at_a_time
    Load visibilities with pyasdm BDFReader.getNDArrays(), which reads only the
    blocks of the selected SPW (and baselines/channels) from the BDF file. When
    False, every subset is loaded with BDFReader.getSubset() (all SPWs) and the SPW
    is selected in memory. Both give the same values.
do_save_blob_info
    Save information about the BDFs loaded (debugging).
"""

use_load_one_spw_at_a_time = True
do_save_blob_info = False
