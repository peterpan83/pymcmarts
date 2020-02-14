'''
@auther Pan Yanqun
@2016.3.27
water-vapor correction
'''
import sys,os
import numpy as np
import utilities as utl


class SensorInfo:
    def __init__(self,sensorName):
        self.sensorName = sensorName
        self.sensorInfoPath = os.path.join(utl.OCSSWDATA+os.sep+sensorName,'msl12_sensor_info.dat')
        # using dictionary save items in sensorinfo
        self.ItemDIc = {}
        self.nwave = 0
        self.wave = []
        self.F0 = []
        self.Tau_r = []
        self.k_oz = []
        self.k_no2 = []
        self.aw = []
        self.bbw = []
        self.ooblw01 = []
        self.ooblw02 = []
        self.ooblw03 = []
        self.t_co2 = []
        self.a_h2o = []
        self.b_h2o = []
        self.c_h2o = []
        self.d_h2o = []
        self.e_h2o = []
        self.f_h2o = []
        self.g_h2o = []
        self.awhite = []
        self.oobwv01 = []
        self.oobwv02 = []
        self.oobwv03 = []
        self.oobwv04 = []
        self.oobwv05 = []
        self.oobwv06 = []
        self.oobwv07 = []
        self.oobwv08 = []
        self.oobwv09 = []
        self.oobwv10 = []
        self.oobwv11 = []
        self.oobwv12 = []


    def readSensorInfo(self,verbose = False):
        if verbose:
            print ('read '+self.sensorInfoPath)
        data = open(self.sensorInfoPath,'rb')
        for row in data:
            if verbose:
                print(row)
            if row[0] == '#' or row[0]==' ' or row[0]=='\n':
                continue

            if row.find('=')<0:
                print(row +' is invalid')
                exit(-1)

            key = row.split('=')[0]
            value = row.split('=')[1]
            if key.find('Nbands')>=0:
                self.nwave = int(value)
                #self.wave = np.zeros(int(value))
            if key.find('Lambda')>=0:
                self.wave.append(float(value))

            if key.find('F0')>=0:
                self.F0.append(float(value))

            if key.find('Tau_r')>=0:
                self.Tau_r.append(float(value))

            if key.find('k_oz')>=0:
                self.k_oz.append(float(value))

            if key.find('k_no2')>=0:
                self.k_no2.append(float(value))

            if key.find('aw')>=0:
                self.aw.append(float(value))

            if key.find('bbw')>=0:
                self.bbw.append(float(value))



            if key.find('ooblw01')>=0:
                self.ooblw01.append(float(value))

            if key.find('ooblw02')>=0:
                self.ooblw02.append(float(value))

            if key.find('ooblw03')>=0:
                self.ooblw03.append(float(value))

            if key.find('t_co2')>=0:
                self.t_co2.append(float(value))




            if key.find('a_h2o')>=0:
                self.a_h2o.append(float(value))

            if key.find('b_h2o')>=0:
                self.b_h2o.append(float(value))

            if key.find('c_h2o')>=0:
                self.c_h2o.append(float(value))

            if key.find('d_h2o')>=0:
                self.d_h2o.append(float(value))

            if key.find('e_h2o')>=0:
                self.e_h2o.append(float(value))

            if key.find('f_h2o')>=0:
                self.f_h2o.append(float(value))

            if key.find('g_h2o')>=0:
                self.g_h2o.append(float(value))



            if key.find('awhite')>=0:
                self.awhite.append(float(value))

            if key.find('oobwv01')>=0:
                self.oobwv01.append(float(value))

            if key.find('oobwv02')>=0:
                self.oobwv02.append(float(value))

            if key.find('oobwv03')>=0:
                self.oobwv03.append(float(value))

            if key.find('oobwv04')>=0:
                self.oobwv04.append(float(value))

            if key.find('oobwv05')>=0:
                self.oobwv05.append(float(value))

            if key.find('oobwv06')>=0:
                self.oobwv06.append(float(value))

            if key.find('oobwv07')>=0:
                self.oobwv07.append(float(value))

            if key.find('oobwv08')>=0:
                self.oobwv08.append(float(value))

            if key.find('oobwv09')>=0:
                self.oobwv09.append(float(value))

            if key.find('oobwv10')>=0:
                self.oobwv10.append(float(value))

            if key.find('oobwv11')>=0:
                self.oobwv11.append(float(value))

            if key.find('oobwv12')>=0:
                self.oobwv12.append(float(value))
        if verbose:
            print ('read sensorInfo completed!\n')


    def aeroob(self,iw,cf,wv,airmass):
        f = (self.oobwv01[iw] + self.oobwv02[iw]*airmass + cf*(self.oobwv03[iw] + self.oobwv04[iw]*airmass))\
            + (self.oobwv05[iw] + self.oobwv06[iw]*airmass + cf*(self.oobwv07[iw] + self.oobwv08[iw]*airmass))*wv\
            + (self.oobwv09[iw] + self.oobwv10[iw]*airmass + cf*(self.oobwv11[iw] + self.oobwv12[iw]*airmass))*wv*wv
        return f