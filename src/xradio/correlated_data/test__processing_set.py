import unittest
from xradio.src.xradio.correlated_data.processing_set import processing_set


class TestProcessingSet(unittest.TestCase):

    def setUp(self):
        # Create a sample processing set
        self.ps = processing_set()
        self.ps["ms1"] = {
            "partition_info": {
                "obs_mode": "OBSERVE_TARGET#ON_SOURCE",
                "spectral_window_name": "spw1",
            },
            "polarization": ["RR", "LL"],
            "visibility": "vis1",
            "frequency": [1e9, 2e9],
        }
        self.ps["ms2"] = {
            "partition_info": {
                "obs_mode": "OBSERVE_TARGET#CALIBRATE_POLARIZATION",
                "spectral_window_name": "spw2",
            },
            "polarization": ["RR"],
            "spectrum": "spec1",
            "frequency": [2e9, 3e9],
        }
        self.ps["ms3"] = {
            "partition_info": {
                "obs_mode": "OBSERVE_TARGET#ON_SOURCE",
                "spectral_window_name": "spw1",
            },
            "polarization": ["LL"],
            "visibility": "vis2",
            "frequency": [3e9, 4e9],
        }

    def test_summary(self):
        # Test the summary method
        summary = self.ps.summary()
        self.assertEqual(
            len(summary), 3
        )  # Check the number of rows in the summary table

    def test_get_ps_max_dims(self):
        # Test the get_ps_max_dims method
        max_dims = self.ps.get_ps_max_dims()
        self.assertEqual(max_dims, {"frequency": 2, "polarization": 2})

    def test_get_ps_freq_axis(self):
        # Test the get_ps_freq_axis method
        freq_axis = self.ps.get_ps_freq_axis()
        self.assertEqual(len(freq_axis), 4)  # Check the length of the frequency axis

    def test_sel(self):
        # Test the sel method
        subset = self.ps.sel(
            obs_mode="OBSERVE_TARGET#ON_SOURCE", polarization=["RR", "LL"]
        )
        self.assertEqual(len(subset), 2)  # Check the number of MSs in the subset

    def test_ms_sel(self):
        # Test the ms_sel method
        subset = self.ps.ms_sel(obs_mode="OBSERVE_TARGET#ON_SOURCE", polarization="RR")
        self.assertEqual(len(subset), 1)  # Check the number of MSs in the subset

    def test_ms_isel(self):
        # Test the ms_isel method
        subset = self.ps.ms_isel(obs_mode="OBSERVE_TARGET#ON_SOURCE", polarization="LL")
        self.assertEqual(len(subset), 1)  # Check the number of MSs in the subset


if __name__ == "__main__":
    unittest.main()
