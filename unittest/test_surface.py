import unittest
from utils import AhmadAerosolReader as AReader
from utils import SciatranReader as SReader
from scipy import integrate
from inputs import input as ip
import numpy as np


class TestSurface(unittest.TestCase):

    def setUp(self):
        self.sensor = 'msia'
        self.bands = AReader.get_bands(self.sensor)
        self.bands_int = np.asarray(self.bands, dtype=int)

    def test_surface(self):
        shape = 'square'
        columns = rows = 64
        wavelenth = self.bands_int[8]
        resolution = 300
        print(wavelenth)
        fname = "../out/mdlsfc_%s_%d_%d_%d"%(shape,columns,resolution,wavelenth)
        ip.surface(fname,0,0.11,rows=rows,columns=columns,shape=shape,resolution=resolution)
