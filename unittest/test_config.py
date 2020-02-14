import unittest
from utils import AhmadAerosolReader as AReader
from utils import SciatranReader as SReader
from scipy import integrate
from inputs import input as ip
from inputs.config import Config
import numpy as np
from scipy import interpolate as inp


class TestAerosol(unittest.TestCase):
    def setUp(self):
        self.sensor = 'msia'
        self.bands = AReader.get_bands(self.sensor)
        self.bands_int = np.asarray(self.bands, dtype=int)

    def test_config(self):
        bandindex = 8
        confg = Config(self.sensor,bandindex)
        confg.set_pixels(64,64)
        confg.set_geometry(30,0,0,0)
        confg.set_resoluztion(300.0,300.0)
        confg.set_surface("msf_%d"%self.bands_int[bandindex])
        confg.set_aerosol_phases("mdlphs_%d"%self.bands_int[bandindex])

        ## get altitude grid from aerosol profile
        aerosolp = SReader.aerosol_profile(taua=0.1)
        z = aerosolp['z'].values #km
        z = z*1000 #convert it from km to m
        # z_str = list(map(lambda x:"%.6E"%(x)),z)                                                                              , z))
        z_str = list(map(lambda x:"%.6E"%(x), z))
        confg.set_altitude_grid(z_str)


        ## get aerosol extinction profile
        referindex = 8 #set 865 as the reference
        angstroms = AReader.get_angstroms(self.sensor,10, referindex, self.bands)
        aprofile_s = ip.aersol(self.bands,angstroms, profile=aerosolp)
        aerosol_extinct = aprofile_s['extinct_%d'%self.bands_int[bandindex]].values
        aerosol_extinct = aerosol_extinct*0.001                                         #covert it from km^-1 into m^-1

        taua_scaled = integrate.cumtrapz(aerosol_extinct, z, initial=0)[-1]
        # print(taua_scaled)
        assert (round(taua_scaled,1)==0.1)



        confg.set_aerosol(map(lambda x:"%.6E"%(x), aerosol_extinct), ['0.9767' for i in range(aerosol_extinct.shape[0])], 38)


        ## get atmosphere profile
        atm_profile = SReader.atm_profile()
        ray_df,unit = ip.rayleight(atm_profile,self.bands)
        ray_df.to_csv("../out/extinction_rayleigh_(%s).csv"%(unit))
        z_ray = ray_df['z[km]'].values*1e3                                               #convert from km to m
        ray_extinct = ray_df['extinction_coe_%d'%self.bands_int[bandindex]].values * 1e2 #convert from cm^-1 it into m^-1
        f = inp.interp1d(z_ray, ray_extinct)
        ray_extinct = f(z)
        confg.set_rayleigh(map(lambda x:"%.6E"%(x), ray_extinct), ['1.000' for i in range(ray_extinct.shape[0])])

        ray_scaled = integrate.cumtrapz(ray_extinct, z, initial=0)[-1]
        # print(taua_scaled)
        # assert (round(ray_scaled,1)==0.1)
        print(ray_scaled)

        ## get temperature from atmosphere profile
        temp_k = ray_df['t[K]'].values
        f = inp.interp1d(z_ray, temp_k)
        temp_k = f(z)
        confg.set_tempprofile(map(lambda x:"%.6E"%(x), temp_k))


        ## get gas extinction profile
        gas_profile = SReader.gas_profile()
        no2_crosssect = SReader.no2_cross_section()
        o3_crosssect = SReader.ozone_cross_section()
        gas_abs,unit = ip.gas_absorption(bands=self.bands,profile_df=gas_profile,no2=no2_crosssect,o3=o3_crosssect)
        gas_abs.to_csv("../out/gas_abs_(%s).csv"%unit)

        z_gas = gas_abs['Z'].values*1e3               #convert from km to m
        o3_extinct = gas_abs['abs_o3_%d' % self.bands_int[bandindex]].values * 1e2 #convert it into m^-1
        f = inp.interp1d(z_gas, o3_extinct)
        o3_extinct = f(z)


        no2_extinct = gas_abs['abs_no2_%d' % self.bands_int[bandindex]].values * 1e2 #convert it into m^-1
        f = inp.interp1d(z_gas, no2_extinct)
        no2_extinct = f(z)

        confg.set_gas([o3_extinct,no2_extinct])

        confg.to_file("../out/config_%d"%(self.bands_int[bandindex]))


if __name__ == '__main__':
    unittest.main()







