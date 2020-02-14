import unittest
from utils import AhmadAerosolReader as AReader
from utils import SciatranReader as SReader
from scipy import integrate
from inputs import input as ip
import numpy as np


class TestRayAndGAS(unittest.TestCase):
    def setUp(self):
        self.sensor = 'msia'
        self.bands = AReader.get_bands(self.sensor)
        self.bands_int = np.asarray(self.bands, dtype=int)

    @unittest.skip("already")
    def test_rayextinct_profile(self):
        atm_profile = SReader.atm_profile()
        ray_df,unit = ip.rayleight(atm_profile,self.bands)
        ray_df.to_csv("../out/extinction_rayleigh_(%s).csv"%(unit))
        print(ray_df.shape)

    def test_gas_profile(self):
        gas_profile = SReader.gas_profile()

        no2_crosssect = SReader.no2_cross_section()
        o3_crosssect = SReader.ozone_cross_section()
        gas_abs,unit = ip.gas_absorption(bands=self.bands,profile_df=gas_profile,no2=no2_crosssect,o3=o3_crosssect)
        gas_abs.to_csv("../out/gas_abs_(%s).csv"%unit)
