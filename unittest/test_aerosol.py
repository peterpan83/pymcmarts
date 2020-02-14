import unittest
from utils import AhmadAerosolReader as AReader
from utils import SciatranReader as SReader
from scipy import integrate
from inputs import input as ip
import numpy as np


class TestAerosol(unittest.TestCase):
    def setUp(self):
        self.sensor = 'msia'
        self.bands = AReader.get_bands(self.sensor)
        self.bands_int = np.asarray(self.bands, dtype=int)

    @unittest.skip("already")
    def test_phases(self):
        result = AReader.get_phase_all("msia", 0)
        assert (len(result)==2)
        assert (result[0].shape[0]==75 and result[1].shape==(80,75))

        for re in result[1]:
            # print(re.sum())
            assert (round(re.sum())==1)

    def test_aeroextinct_profile(self):
        aprofile = SReader.aerosol_profile(taua=0.1)
        taua_scaled = integrate.cumtrapz(aprofile.values[:, 1], aprofile.values[:, 0], initial=0)[-1]
        # print(taua_scaled)
        assert (round(taua_scaled,1)==0.1)
        referindex = 5
        angstroms = AReader.get_angstroms(self.sensor,10, referindex, self.bands)

        aprofile_s = ip.aersol(self.bands,angstroms, profile=aprofile)
        assert (aprofile_s.columns.shape[0]==len(self.bands)+2)



if __name__ == '__main__':
    unittest.main()