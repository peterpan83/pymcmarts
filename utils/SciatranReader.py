import os
import pandas as pd
from collections import defaultdict
from glob import glob
import numpy as np
from scipy import integrate

def read_abs_f(path):
#     o3_ab_f = "/home/pan/SCIATRAN/DATA_BASES/SPECTRA/O3_243K_V2_0.dat"
    data = []
    with open(path) as f:
        for line in f:
            if line.strip().startswith("#"):
                continue
#         print(line.split("   "))
            data.append([ float(c) for c in line.split(" ") if len(c.strip())>0])
        df = pd.DataFrame(data=data,columns=["wavelength","sigma_abs"])
    return df


def ozone_cross_section(absorption_o3_dir = "../database/spectra/o3"):
    '''
    :param absorption_o3_files: o3 absorption cross section database
    "/home/pan/SCIATRAN/DATA_BASES/SPECTRA/O3_*K_V2_0.dat"
    :return: dictionary with K and dataframe as the key and value, respectively
    '''
    absorption_o3_dic = defaultdict()
    absorption_o3_files = glob(os.path.join(absorption_o3_dir, "O3_*K_V2_0.dat"))
    for p in absorption_o3_files:
        path,filename = os.path.split(p)
        # tempreture K
        t = float(filename.split("_")[1][:-1])
        # print(t)
        df = read_abs_f(p)
        absorption_o3_dic[t] = df
    return absorption_o3_dic

def no2_cross_section(absorption_no2_dir = "../database/spectra/no2"):
    '''
    :param absorption_no2_files: no2 absorption cross section database
    "/home/pan/SCIATRAN/DATA_BASES/SPECTRA/NO2_*K.dat"
    :return:
    '''
    absorption_no2_dic = defaultdict()
    absorption_no2_files = glob(os.path.join(absorption_no2_dir,"NO2_*K.dat"))
    for p in absorption_no2_files:
        path, filename = os.path.split(p)
        t = float(os.path.splitext(filename)[0].split("_")[1][:-1])
        df = read_abs_f(p)
        absorption_no2_dic[t] = df
    return absorption_no2_dic

def gas_profile(profile_f="../database/profiles/SCE_ABSORBER.OUT"):
    '''
    read the gas profile data
    :param profile_f: "/home/pan/SCIATRAN/Execute-3.7.1/DATA_OUT/SCE_ABSORBER.OUT"
    :return:
    '''
    data_profile = []
    with open(profile_f,'r') as abs_p:
        for i, line in enumerate(abs_p):
            if i == 9:
                columns = [c.strip() for c in line.split("  ") if len(c.strip()) > 0]
                continue
                #         print(line)
            if (i > 11) & (i < 113):
                #             print(line)
                try:
                    dataline = [float(c.strip()) for c in line.split(" ") if len(c.strip()) > 0]
                    data_profile.append(dataline)
                except Exception as e:
                    raise (e)

    data_profile_df = pd.DataFrame(data=data_profile, columns=columns)
    return data_profile_df


def atm_profile(profile = "../database/profiles/usstandard.dat"):
    '''
    this profile is for Rayleigh extinction
    :param profile: us standard atmosphere profile
    :return: DataFrame ['z','']
    '''
    data = []
    with open(profile,'r') as atm_profile:
        for i, line in enumerate(atm_profile):
            if i < 1:
                continue
            if i == 1:
                columns = line.split(" ")
                columns = [c.strip() for c in columns[1:] if len(c.strip()) > 0]
                print(columns)
                continue
            data.append([float(c.strip()) for c in line.split(" ") if len(c.strip()) > 0])
    atm_profile_df = pd.DataFrame(data=data,columns=columns)
    ## ideal gas law: N = p/(T*R), R =8.2555738E-5 m^3 atm K-1 mol-1
    ## p: 1 standard air pressure
    R = 8.2055738e-5
    standard_pre = 1.013e3
    atm_profile_df['N'] = atm_profile_df.apply(lambda x: x['p[mb]'] / standard_pre / (R * x['t[K]']) * 1e-6 * 6.02214076e23, axis=1)  # cm^-3
    return atm_profile_df


def aerosol_profile(profile = "../database/profiles/scia_aer.inp", taua=0.01):
    refer_wv_flag, profile_flag = -1, -1
    refer_wv = 0.0
    profiles = 0
    profile_data = None
    with open(profile,'r') as scia_aer:
        for i, line in enumerate(scia_aer):
            if line.startswith("#"):
                continue
                #         print(i,line)
            if "Reference wavelength" in line:
                refer_wv_flag = 1
                continue
            if refer_wv_flag == 1:
                refer_wv = float(line.strip())
                refer_wv_flag = -1
                continue
            if "Profile" == line.strip():
                profile_flag = 1
                continue
            if profile_flag == 1:
                if profiles == 0:
                    # print(line)
                    profiles = int(line.strip())
                    profile_data = np.zeros((profiles, 2), float)
                    j = 0
                    continue
                elif (j < 33):
                    line_re = line.split("#")[0]
                    height, extection_c = float(line_re.split(",")[0].strip()), float(line_re.split(",")[1].strip())
                    profile_data[j] = np.array([height, extection_c])
                    # print(j, height, extection_c)
                    j += 1

    ## adjust
    profile_data[0:3, 1] = np.array([0.004, 0.006, 0.01])
    taua_origin = integrate.cumtrapz(profile_data[:,1],profile_data[:,0],initial=0)[-1]
    profile_data[:,1] = profile_data[:,1]*(taua/taua_origin)
    return pd.DataFrame(data=profile_data,columns=['z','extinct'])











