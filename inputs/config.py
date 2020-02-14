from collections import defaultdict, OrderedDict
from utils.aerosolmodle import sensorInfo as SI


class Config():

    def __init__(self,sensor,bandindex):
        self.si = SI.SensorInfo(sensorName=sensor)
        self.si.readSensorInfo()

        #------------------SRC-------------------------------------#
        src_dic = OrderedDict()
        src_dic['jnorm'] = 0
        src_dic['nsrc'] = 1
        src_dic['fsrc'] = self.si.F0[bandindex] *10 #extresstial irradiance ()
        src_dic['dqsrc'] = 0.2665666
        src_dic['dwlen'] = 0.0
        src_dic['jphisrc'] = 0
        src_dic['jdefsrc'] = 1
        src_dic['thesrc'] = 120.0 # solar angle = 180 - solar zenith angel, default 120
        src_dic['phisrc'] = 0 # solar azimuth angel, defaut 0

        #---------------TEC---------------------------------------#
        tec_dic = OrderedDict()
        tec_dic['nbyte'] = 4
        tec_dic['nstrun'] = 0
        tec_dic['nsmax'] = 100000
        tec_dic['nsss'] = 1
        tec_dic['jtech'] = 1

        #---------------INTEG-------------------------------------#
        integ_dic = OrderedDict()
        integ_dic['jsflx'] = 0
        integ_dic['jshrt'] = 0
        integ_dic['nrdc'] = 1
        integ_dic['nxr'] = 64
        integ_dic['nyr'] = 64
        integ_dic['therdc'] = 0.0
        integ_dic['phirdc'] = 0.0
        integ_dic['jzrdc'] = 100
        integ_dic['ncam'] = 0
        integ_dic['npot'] = 0

        #-----------------MODEL-------------------------------#
        model_dic = OrderedDict()
        model_dic['jrfr'] = 0
        model_dic['xbin'] = 300.0  #resolution
        model_dic['ybin'] = 300.0  #resolution
        model_dic['nx'] = 64       #pixels
        model_dic['ny'] = 64       #pixels
        model_dic['jsph'] = 0
        model_dic['nz'] = 32       #layers of atmosphere
        model_dic['jtmplev'] = 1

        model_dic['zgrd'] = []    #altitude grid
        model_dic['tmpa1d'] = []  #temperture
        model_dic['ext1st'] = []  #extinction coefficient of aerosol
        model_dic['omg1st'] = []  #transmittance
        model_dic['jpf1st'] = []  #aerosol model (relative phase function)

        model_dic['ext2nd'] = [] #extinction coefficient of rayleigh
        model_dic['omg2nd'] = [] #transmittance
        model_dic['jpf2nd'] = [] #rayleigh phase

        model_dic['nkd'] = 1
        model_dic['wkd'] = '1.000000E+00,'

        model_dic['absg1d'] = []

        #----------------- MDLPHS-----------------------------#
        mdlphs_dic = OrderedDict()
        mdlphs_dic['filephs'] = ""

        #--------------MDLSFC--------------------------------#
        mdlsfc_dic = OrderedDict()
        mdlsfc_dic['filesfc'] = ""
        mdlsfc_dic['nxb'] = 64
        mdlsfc_dic['nyb'] = 64

        #----------------MDLA3D------------------------------#
        mdla3d_dic = OrderedDict()

        self.models = OrderedDict()
        self.models['SRC'] = src_dic
        self.models['TECH'] = tec_dic
        self.models['INTEG'] = integ_dic
        self.models['MODEL'] = model_dic
        self.models['MDLPHS'] = mdlphs_dic
        self.models['MDLSFC'] = mdlsfc_dic
        self.models['MDLA3D'] = mdla3d_dic


    def __write(self,pf,dic):
        for key, value in dic.items():
            # print(key)
            if key == 'absg1d':
                for i, gas in enumerate(value):
                    for j, extinct in enumerate(gas):
                        pf.write(" absg1d( %d,%d) =  %.6E,\n"%(j+1, i+1, extinct))
                continue

            if type(value) == list:
                s = ", ".join(value)
                pf.write(" %s = %s\n" % (key, s))
                continue


            pf.write(" %s = %s\n"%(key,str(value)))

    def to_file(self,pfile):
        with open(pfile,'w') as pf:
            for key, value in self.models.items():
                pf.write("&%s\n"%(key.lower()))
                self.__write(pf,value)
                pf.write("/\n\n")

    def set_geometry(self,theta_s,theta_v,phi_s, phi_v):
        '''
        :param theta_s: solar zenith angel
        :param theta_v: veiwing zenith angel
        :param relaz_s: solar azimuth
        :param relaz_v: sensor azimuth
        :return: NONE
        '''
        self.models['SRC']['thesrc'] = 180-theta_s
        self.models['SRC']['phisrc'] = phi_s
        self.models['INTEG']['therdc'] = theta_v
        self.models['INTEG']['phirdc'] = phi_v

    def set_resoluztion(self,xbin,ybin):
        '''
        set resolution in x and y direction,respectively. in units of meters
        :param xbin:
        :param ybin:
        :return:
        '''
        self.models['MODEL']['xbin'] = xbin
        self.models['MODEL']['ybin'] = ybin

    def set_pixels(self,rows,columns):
        '''set the dimention of the output
        :param rows:
        :param columns:
        :return:
        '''
        self.models['MODEL']['nx'] = columns
        self.models['MODEL']['ny'] = rows
        self.models['INTEG']['nxr'] = columns
        self.models['INTEG']['nyr'] = rows
        self.models['MDLSFC']['nxb'] = columns
        self.models['MDLSFC']['nyb'] = rows


    def set_altitude_grid(self, altitude_grid):
        self.models['MODEL']['zgrd'] = altitude_grid
        self.__set_number_grid(len(altitude_grid)-1)

    def __set_number_grid(self,nz):
        self.models['MODEL']['nz'] = nz

    def set_tempprofile(self, temprature_profile):
        self.models['MODEL']['tmpa1d'] = temprature_profile

    def set_gas(self,gas_absorption):
        '''
        :param gas_absorption:  gas absorption profiles
        :return:
        '''
        self.models['MODEL']['absg1d'] = gas_absorption

    def set_rayleigh(self, extinction_coe,transmittance):
        self.models['MODEL']['ext2nd'] = extinction_coe
        self.models['MODEL']['omg2nd'] = transmittance
        self.models['MODEL']['jpf2nd'] = ['0' for i in range(self.models['MODEL']['nz']+1)]

    def set_aerosol(self, extinction_coe,transmittance,phase):
        self.models['MODEL']['ext1st'] = extinction_coe
        self.models['MODEL']['omg1st'] = transmittance
        self.models['MODEL']['jpf1st'] = [str(phase) for i in range(self.models['MODEL']['nz']+1)]


    def set_aerosol_phases(self,phasefile):
        self.models['MDLPHS']['filephs'] = "'%s'"%phasefile

    def set_surface(self,surfacefile):
        self.models['MDLSFC']['filesfc'] = "'%s'"%surfacefile




def gen_surface(data,outputf):
    pass


def gen_main_config():
    pass
