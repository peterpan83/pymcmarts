from pandas import DataFrame
import numpy as np
from utils.aerosolmodle import utilities

def gas_absorp_crosssection(t, gas_dic, wv=500):
    '''
    get absorption cross section
    t: tempture (K)
    wv: wavelength (nm)
    '''
    # if gas == 'o3':
    #     gas_dic = absorption_o3_dic
    #
    # elif gas == "no2":
    #     gas_dic = absorption_o3_dic
    #
    # gas_dic = absorption_o3_dic
    assert (isinstance(gas_dic,dict))
    t_s = gas_dic.keys()
    t_s = sorted(t_s)

    t_l, t_b = 0, 5000
    if t > t_s[-1]:
        t_b = t_s[-1]
        t_l = t_b
        ratio = 1
    elif t < t_s[0]:
        t_l = t_s[0]
        t_b = t_l
        ratio = 0
    else:
        for t_0, t_1 in zip(t_s[:-1], t_s[1:]):
            if (t >= t_0) & (t < t_1):
                t_l, t_b = t_0, t_1
                ratio = 1 - (t - t_l) / (t_b - t_l)

    df_temp_1 = gas_dic[t_b]
    #     print(df_temp_1)

    #     print(df_temp_1.where((df_temp_1['wavelength']>=wv)))
    # v1 = df_temp_1.where((df_temp_1['wavelength'] >= (wv - 1.0)) & (df_temp_1['wavelength'] < (wv + 1.0))).mean()[
    #     'sigma_abs']

    # df_temp_1['wavelength']

    index =  utilities.windex(wv,df_temp_1['wavelength'].values,df_temp_1['wavelength'].shape[0])
    # print(index)
    v1 = df_temp_1.iloc[index]['sigma_abs']


    df_temp_2 = gas_dic[t_l]
    index = utilities.windex(wv, df_temp_2['wavelength'].values, df_temp_2['wavelength'].shape[0])
    # v2 = df_temp_2.where((df_temp_2['wavelength'] >= (wv - 1.0)) & (df_temp_2['wavelength'] < (wv + 1.0))).mean()[
    #     'sigma_abs']
    v2 = df_temp_2.iloc[index]['sigma_abs']

    #     print(t_l,t_b,v1,v2,ratio,v1*ratio + v2*(1-ratio))
    #     print(t_s)
    return v1 * ratio + v2 * (1 - ratio)



def gas_absorption(profile_df, bands, **kwargs):
    '''
    generate gas absorption profile, based on the concentration profile and cross section
    :param profile_df: gas concentration profile
    :param bands: bands
    :param kwargs: key->gas name, value->cross section dict
    :return: gas absorption profile
    '''
    assert (isinstance(profile_df,DataFrame))

    for gas_name, cross_section in kwargs.items():
        print (gas_name)
        for w in bands:
            profile_df['sigma_abs_%s_%d' % (gas_name,w)] = profile_df.apply(lambda z: gas_absorp_crosssection(t=z['T'], gas_dic=cross_section, wv=w), axis=1)
            # profile_df['sigma_abs_no2_%d' % w] = profile_df.apply(lambda z: get_cross_section(t=z['T'], gas='no2', wv=w), axis=1)
            profile_df['abs_%s_%d' % (gas_name,w)] = profile_df['sigma_abs_%s_%d' % (gas_name,w)] * profile_df[gas_name]
            # profile_df['abs_no2_%d' % w] = profile_df['sigma_abs_o3_%d' % w] * profile_df['no2']
    # profile_df.to_csv('../out/gas_abs_(cm-1).csv')
    return profile_df,"cm-1"

def aersol(bands,angstroms, profile):
    '''
    :param bands:
    :param angstroms:
    :param profile: DataFrame ['z','extinct']
    :return: profile with all bands
    '''
    for i, w in enumerate(bands):
        angstrom = angstroms[i]
        profile['extinct_%d'%w] = profile.apply(lambda x:x['extinct']*angstrom,axis=1)
        # print(profile_extinction_c_dic[w])

    return profile
    # profile.to_csv('extinction_aerosol_(km-1).csv')

def rayleight(profile_df, bands):
    '''
    calculate Rayleight scattering extinction profile
    :param profile_df: atmosphere profile
    :param bands: bands
    :param kwargs:
    :return:DataFrame, unit
    '''
    ## this is from sciatran SCE_RAYLEIGH.OUT, scatteering cross section from molecule
    lambda0 = 550
    sigma_lambda0 = 4.513E-27  # cm^2/sr
    simgas_lambda = [(lambda0 * 1.0 / l) ** 4 * sigma_lambda0 for l in bands] #rayleigh scattering law
    for i, w in enumerate(bands):
        profile_df['extinction_coe_%d' % w] = profile_df.apply(lambda x: x['N'] * simgas_lambda[i], axis=1)
        # print(integrate.cumtrapz(profile_df['extinction_coe_%d' % w] * 1e5)[-1])
        #
        # plt.plot(atm_profile_df['z[km]'], atm_profile_df['extinction_coe_%d' % w], 'o-', label="%dnm" % w)
    # plt.xlabel("height (km)", fontsize=16)
    # plt.ylabel("extinction coefficient ($km^{-1}$)", fontsize=16)
    # plt.legend()

    # profile_df.to_csv("../out/extinction_rayleigh_(cm-1).csv")
    return profile_df,'cm-1'


def surface(fname, rhow,rhol,rows=64,columns=64,shape='square',resolution=300):

    image = np.zeros((rows,columns),dtype=float)
    watermask = np.full_like(image,False,dtype=np.bool)

    if shape == 'square':
        image[rows//4:rows//4*3,columns//4:columns//4*3] = rhow
        watermask[rows//4:rows//4*3,columns//4:columns//4*3] = True
        image[~watermask] = rhol
        # print(np.any(image>0))
        # print(shape)

    if shape == 'river':
        image[:,columns//4:columns//4*3] = rhow
        watermask[:,columns//4:columns//4*3] = True
        image[~watermask] = rhol

    # with open("/home/pan/3DMonterCarlo/mcarats-0.9.5/examples/case2/mdlsfc_%d_256" % w, 'w') as f:
    with open(fname,'w') as f:
        f.write("%mdlsfc\n")
        f.write("# tmps2d\n")
        f.write("%d*300.0\n" % (rows * columns))
        f.write("# jsfc2d\n")
        f.write("%d*1\n" % (rows * columns))
        f.write("# psfc2d\n")
        image = image.flatten()
        for i, v in enumerate(image):
            f.write("  %.6f" % v)
            if i % 10 == 0:
                f.write("\n")
        f.write("\n")

def gen_config():
    pass


def gen_aerosol_phase(sensor, band,scatt,phases,**kwargs):
    '''
    :param sensor:
    :param band:
    :param scatt:
    :param phases:
    :param kwargs:
    :return:
    '''
    if "rhs" in kwargs:
        rhs = kwargs['rhs']
    if "fmfs" in kwargs:
        fmfs = kwargs['fmfs']

    with open("../out/mdlphs_%d"%band, "w") as f:
        f.write("%mdlphs\n")
        f.write(" %d : # of definitions \n" % (80))
        for i in range(80):
            f.write("##\n")
            f.write(" %d : # of angles rh=%d, fmf=%d\n" % (scatt.shape[0], rhs[i], fmfs[i]))
            f.write("# ang, phs\n")
            for s, phs in zip(scatt, phases[i,:]):
                f.write("  %.2f,  %.6f\n" % (s, phs))

    print("../out/mdlphs_%d"%band)