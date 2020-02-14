import pandas as pd
import numpy as np
import os
import unittest
from utils import AhmadAerosolReader as AReader
from utils import SciatranReader as SReader
from scipy import integrate
from inputs import input as ip
from inputs.config import Config
from scipy import interpolate as inp


taua_nirls = [0.005,0.010,0.050,0.100, 0.500]
aero_phs = [0,9,20,29,50,59,70,79]
resolutions = [30.0,100.0,500.0,1500.0,5000.0]
theta_s = [5,10,30,50]

rho_diff = []

rho_diff_df = pd.read_csv("../database/s2_rho_diff.csv", index_col=0)
sensor = 'msia'
bands = AReader.get_bands(sensor)
bands_int = np.asarray(bands, dtype=int)

shape = "river"
columns = rows = 64



for bandindex, band in enumerate(rho_diff_df.columns[:-3]):
    confg = Config(sensor, bandindex)
    confg.set_pixels(rows, columns)

    if not os.path.exists("../out/%d"%(bands_int[bandindex])):
        os.mkdir("../out/%d"%(bands_int[bandindex]))

    cur_dir = "../out/%d"%(bands_int[bandindex])
    wavelenth = bands_int[bandindex]

    diff_25, diff_75 = rho_diff_df.loc['25%'][band], rho_diff_df.loc['75%'][band]

    for taua_index, taua in enumerate(taua_nirls):
        ## get altitude grid from aerosol profile
        aerosolp = SReader.aerosol_profile(taua=taua)
        z = aerosolp['z'].values  # km
        z = z * 1000  # convert it from km to m
        # z_str = list(map(lambda x:"%.6E"%(x)),z)                                                                              , z))
        z_str = list(map(lambda x: "%.6E" % (x), z))
        confg.set_altitude_grid(z_str)

        ## get atmosphere profile
        atm_profile = SReader.atm_profile()
        ray_df, unit = ip.rayleight(atm_profile, bands)
        ray_df.to_csv("../out/extinction_rayleigh_(%s).csv" % (unit))
        z_ray = ray_df['z[km]'].values * 1e3  # convert from km to m
        ray_extinct = ray_df['extinction_coe_%d' % bands_int[
            bandindex]].values * 1e2  # convert from cm^-1 it into m^-1
        f = inp.interp1d(z_ray, ray_extinct)
        ray_extinct = f(z)
        confg.set_rayleigh(map(lambda x: "%.6E" % (x), ray_extinct),
                           ['1.000' for i in range(ray_extinct.shape[0])])

        ray_scaled = integrate.cumtrapz(ray_extinct, z, initial=0)[-1]
        # print(taua_scaled)
        # assert (round(ray_scaled,1)==0.1)
        print(ray_scaled)

        ## get temperature from atmosphere profile
        temp_k = ray_df['t[K]'].values
        f = inp.interp1d(z_ray, temp_k)
        temp_k = f(z)
        confg.set_tempprofile(map(lambda x: "%.6E" % (x), temp_k))

        ## get gas extinction profile
        gas_profile = SReader.gas_profile()
        no2_crosssect = SReader.no2_cross_section()
        o3_crosssect = SReader.ozone_cross_section()
        gas_abs, unit = ip.gas_absorption(bands=bands, profile_df=gas_profile, no2=no2_crosssect,
                                          o3=o3_crosssect)
        gas_abs.to_csv("../out/gas_abs_(%s).csv" % unit)

        z_gas = gas_abs['Z'].values * 1e3  # convert from km to m
        o3_extinct = gas_abs['abs_o3_%d' % bands_int[bandindex]].values * 1e2  # convert it into m^-1
        f = inp.interp1d(z_gas, o3_extinct)
        o3_extinct = f(z)

        no2_extinct = gas_abs['abs_no2_%d' % bands_int[bandindex]].values * 1e2  # convert it into m^-1
        f = inp.interp1d(z_gas, no2_extinct)
        no2_extinct = f(z)

        confg.set_gas([o3_extinct, no2_extinct])

        for theta in theta_s:
            for model_index in aero_phs:
                ## get aerosol extinction profile
                referindex = 8  # set 865 as the reference
                angstroms = AReader.get_angstroms(sensor, model_index, referindex, bands)
                aprofile_s = ip.aersol(bands, angstroms, profile=aerosolp)
                aerosol_extinct = aprofile_s['extinct_%d' % bands_int[bandindex]].values
                aerosol_extinct = aerosol_extinct * 0.001  # covert it from km^-1 into m^-1

                taua_scaled = integrate.cumtrapz(aerosol_extinct, z, initial=0)[-1]
                print(taua_scaled)
                # assert (round(taua_scaled, 3) == taua)

                confg.set_aerosol(map(lambda x: "%.6E" % (x), aerosol_extinct),
                                  ['0.9767' for i in range(aerosol_extinct.shape[0])], model_index)

                for res in resolutions:

                    rho_diff = np.linspace(diff_25, diff_75, 8)

                    for index_diff, diff in enumerate(rho_diff):
                        if diff < 0:
                            rhow = abs(diff)
                            rhol = 0.0
                        else:
                            rhow = 0
                            rhol = abs(diff)

                        confg.set_geometry(theta, 0, 0, 0)
                        confg.set_resoluztion(res, res)

                        fpath = os.path.join(cur_dir,"mdlsfc_%s_%d_%d_%d_%d" % (shape, columns, res, wavelenth,index_diff))
                        fname = os.path.split(fpath)[-1]

                        ##
                        ip.surface(fpath, rhow=rhow, rhol=rhol, rows=rows, columns=columns, shape=shape, resolution=res)

                        confg.set_surface(fname)
                        confg.set_aerosol_phases("mdlphs_%d" % wavelenth)


                        confg.to_file(os.path.join( cur_dir,"config_%s_%d_%d_%d_%d_%d_%d_%d" % (shape,model_index,theta,taua_index,
                                                                                             columns, res, wavelenth,index_diff)))