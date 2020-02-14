import unittest
from utils import AhmadAerosolReader as AReader
from utils import SciatranReader as SReader
from scipy import integrate
from inputs import input as ip
from inputs.config import Config
import numpy as np
from scipy import interpolate as inp


class TestGas(unittest.TestCase):

    def setUp(self):
        self.sensor = 'msia'
        self.bands = AReader.get_bands(self.sensor)
        self.bands_int = np.asarray(self.bands, dtype=int)

    def test_gas(self):
        gas_profile = SReader.gas_profile()
        no2_crosssect = SReader.no2_cross_section()
        o3_crosssect = SReader.ozone_cross_section()
        gas_abs,unit = ip.gas_absorption(bands=self.bands,profile_df=gas_profile,no2=no2_crosssect,o3=o3_crosssect)
        gas_abs.to_csv("../out/gas_abs_(%s).csv"%unit)




