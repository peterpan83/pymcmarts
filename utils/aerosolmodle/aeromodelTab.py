'''
@auther Pan Yanqun
@2016.3.18
'''

import os,sys
import numpy as np
from pyhdf.SD import *
from scipy import interpolate

import aeromodle as am
import constant as cns
import utilities as ult
import sensorInfo as SI
from scipy.interpolate import RegularGridInterpolator

# AeromodelTab is used for loading aerosol models
class AeromodelTab:
    nw = 1.334
    firstCall_aeroob_cf = 1
    iw1_aeroob_cf = 0
    iw2_aeroob_cf = 0
    firstCall_transmittance = 1
    intexp = None
    inttst = None

    def __init__(self,sensorName,sensorinfo,ocsswdata='/home/pan/ocssw/run/data',nmodels=80,modelnames=''):
        if sensorinfo==None:
            self.si = SI.SensorInfo(sensorName=sensorName)
            self.si.readSensorInfo()
        else:
            self.si = sensorinfo

        self.rhs = []
        self.fmfs = []

        self.modelDic = {}

        self.sensorName=sensorName
        self.nwave=0
        self.nmodel=nmodels
        self.nsolz=0
        self.nsenz=0
        self.nphi=0
        self.nscatt=0
        self.dtran_nwave=0
        self.dtran_ntheta=0

        self.wave=np.array([])
        self.solz=np.array([])
        self.senz=np.array([])
        self.phi=np.array([])
        self.scatt=np.array([])

        #se transmittance spectral bands and angles
        self.dtran_wave=np.array([])
        self.dtran_theta=np.array([])
        self.dtran_airmass=np.array([])

        self.aeromodels=np.array([])

        self.aerosolpath=os.path.join(ocsswdata+os.path.sep+sensorName,'aerosol')
        self.modelnames=modelnames
        self.models=[]

        self.openSD=None
        self.openSubSD=None

        self.mindex1 = None
        self.mindex2 = None
        self.wt = 0
        if sensorName == 'oli':
            self.aotBandIndex = 4
            self.aotWave = 865
        elif sensorName == 'goci':
            self.aotBandIndex = 7
            self.aotWave = 865


    def ss2ms_coef(self,im,solz,senz,phi):
        tempModel = self.models[im]
        temp = np.zeros((len(self.wave),4))
        temp[:,0] = self.wave
        temp[:,1] = solz
        temp[:,2] = phi
        temp[:,3] = senz

        a_coef = self.acostInterpolator(temp)
        b_coef = self.bcostInterpolator(temp)
        c_coef = self.ccostInterpolator(temp)
        return (a_coef,b_coef,c_coef)


    #return coefficients of function relating single
    # scattering to multiple scattering at the input geometry
    def ss2ms_coef_old(self,im,solz, senz, phi):
        tempModel = self.models[im]
        a_coef = np.zeros(self.nwave)
        b_coef = np.zeros(self.nwave)
        c_coef = np.zeros(self.nwave)
        #solz
        for i in range(self.nsolz):
            if solz < self.solz[i]:
                break
        isolz1 = max(i-1,0)
        isolz2 = min(i,self.nsolz-1)
        if isolz2 != isolz1:
            r = (solz-self.solz[isolz1])/(self.solz[isolz2]-self.solz[isolz1])
        else:
            r = 0.0

        # senz
        for i in range(self.nsenz):
            if senz < self.senz[i]:
                break
        isenz1 = max(i-1,0)
        isenz2 = min(i,self.nsenz-1)
        if isenz2 != isenz1:
            p = (senz-self.senz[isenz1])/(self.senz[isenz2]-self.senz[isenz1])
        else:
            p = 0.0

        # phi
        phi = abs(phi)
        for i in range(self.nphi):
            if phi < self.phi[i]:
                break
        iphi1 = max(i-1,0)
        iphi2 = min(i,self.nphi-1)
        if iphi2 != iphi1:
            q = (phi-self.phi[iphi1])/(self.phi[iphi2]-self.phi[iphi1])
        else:
            q = 0.0

        if isolz2 == 0:
            for iw in range(self.nwave):
                as000 = tempModel.acost[self.INDEX(iw,isolz1,0,isenz1)]
                as100 = tempModel.acost[self.INDEX(iw,isolz1,0,isenz2)]
                as001 = tempModel.acost[self.INDEX(iw,isolz2,0,isenz1)]
                as101 = tempModel.acost[self.INDEX(iw,isolz2,0,isenz2)]

                ai000 = tempModel.bcost[self.INDEX(iw,isolz1,0,isenz1)]
                ai100 = tempModel.bcost[self.INDEX(iw,isolz1,0,isenz2)]
                ai001 = tempModel.bcost[self.INDEX(iw,isolz2,0,isenz1)]
                ai101 = tempModel.bcost[self.INDEX(iw,isolz2,0,isenz2)]

                ac000 = tempModel.ccost[self.INDEX(iw,isolz1,0,isenz1)]
                ac100 = tempModel.ccost[self.INDEX(iw,isolz1,0,isenz2)]
                ac001 = tempModel.ccost[self.INDEX(iw,isolz2,0,isenz1)]
                ac101 = tempModel.ccost[self.INDEX(iw,isolz2,0,isenz2)]

                # print(str((im,solz, senz, phi)))
                a_coef[iw] = (1.-p)*(1.-r)*as000 + p*r*as101 + (1.-p)*r*as001 + p*(1.-r)*as100
                b_coef[iw] = (1.-p)*(1.-r)*ai000 + p*r*ai101 + (1.-p)*r*ai001 + p*(1.-q)*(1.-r)*ai100
                c_coef[iw] = (1.-p)*(1.-r)*ac000 + p*r*ac101 + (1.-p)*r*ac001 + p*(1.-q)*(1.-r)*ac100
        else:
            for iw in range(self.nwave):
                as000 = tempModel.acost[iw][isolz1][iphi1][isenz1]
                as100 = tempModel.acost[iw][isolz1][iphi1][isenz2]
                as010 = tempModel.acost[iw][isolz1][iphi2][isenz1]
                as110 = tempModel.acost[iw][isolz1][iphi2][isenz2]
                as001 = tempModel.acost[iw][isolz2][iphi1][isenz1]
                as011 = tempModel.acost[iw][isolz2][iphi2][isenz1]
                as101 = tempModel.acost[iw][isolz2][iphi1][isenz2]
                as111 = tempModel.acost[iw][isolz2][iphi2][isenz2]

                ai000 = tempModel.bcost[iw][isolz1][iphi1][isenz1]
                ai100 = tempModel.bcost[iw][isolz1][iphi1][isenz2]
                ai010 = tempModel.bcost[iw][isolz1][iphi2][isenz1]
                ai110 = tempModel.bcost[iw][isolz1][iphi2][isenz2]
                ai001 = tempModel.bcost[iw][isolz2][iphi1][isenz1]
                ai011 = tempModel.bcost[iw][isolz2][iphi2][isenz1]
                ai101 = tempModel.bcost[iw][isolz2][iphi1][isenz2]
                ai111 = tempModel.bcost[iw][isolz2][iphi2][isenz2]

                ac000 = tempModel.ccost[iw][isolz1][iphi1][isenz1]
                ac100 = tempModel.ccost[iw][isolz1][iphi1][isenz2]
                ac010 = tempModel.ccost[iw][isolz1][iphi2][isenz1]
                ac110 = tempModel.ccost[iw][isolz1][iphi2][isenz2]
                ac001 = tempModel.ccost[iw][isolz2][iphi1][isenz1]
                ac011 = tempModel.ccost[iw][isolz2][iphi2][isenz1]
                ac101 = tempModel.ccost[iw][isolz2][iphi1][isenz2]
                ac111 = tempModel.ccost[iw][isolz2][iphi2][isenz2]

                a_coef[iw] = (1.-p)*(1.-q)*(1.-r)*as000 + p*q*r*as111+ p*(1.-q)*r*as101 + (1.-p)*q*(1.-r)*as010+ \
                                 p*q*(1.-r)*as110 + (1.-p)*(1.-q)*r*as001+ (1.-p)*q*r*as011 + p*(1.-q)*(1.-r)*as100

                b_coef[iw] = (1.-p)*(1.-q)*(1.-r)*ai000 + p*q*r*ai111+ p*(1.-q)*r*ai101 + (1.-p)*q*(1.-r)*ai010+ \
                                 p*q*(1.-r)*ai110 + (1.-p)*(1.-q)*r*ai001+ (1.-p)*q*r*ai011 + p*(1.-q)*(1.-r)*ai100

                c_coef[iw] = (1.-p)*(1.-q)*(1.-r)*ac000 + p*q*r*ac111+ p*(1.-q)*r*ac101 + (1.-p)*q*(1.-r)*ac010+ \
                                 p*q*(1.-r)*ac110 + (1.-p)*(1.-q)*r*ac001+ (1.-p)*q*r*ac011 + p*(1.-q)*(1.-r)*ac100

        return (a_coef,b_coef,c_coef)


    #get diffuse transmittance of two directions(solar and sensor)
    #for a specific aerosol model and fixed taua
    def get_difftrans_fixedaot(self,rh,fmf,solz, senz, phi,wv,pr,rhoa,nwave):
        tsol = None
        tsen = None
        wave = self.wave
        #nwave = self.nwave
        mu0 = np.cos(solz/cns.RADEG)
        mu = np.cos(senz/cns.RADEG)
        airmass = 1.0/mu0 + 1.0/mu

        csolz  = np.cos(solz/cns.RADEG)
        csenz  = np.cos(senz/cns.RADEG)
        taua = np.zeros(nwave)

        taur = np.array(self.si.Tau_r)[:nwave]

        if self.mindex1==None and self.mindex2==None:
            mindex1,mindex2,wt = self.getrh_index(rh=rh)
        else:
            mindex1 = self.mindex1
            mindex2 = self.mindex2
            wt = self.wt
        if wt == 0:
            modmin,modmax,modrat = self.getmodel_num(rhindex=mindex1,fmf=fmf)
            if modrat == 0:
                #get SS aerosol reflectance at longest sensor wavelength */
                rhoas = self.rhoa2rhos(modnum=modmin,solz=solz,senz=senz,phi=phi,wv=wv,rhoa=rhoa,wave=wave,nwave=nwave)
                phase  = self.get_phase(im=modmin,solz=solz,senz=senz,phi=phi)
                #get aerosol optical thickness at longest sensor wavelength */
                aotnirl = rhoas[self.aotBandIndex]*(4.0*csolz*csenz)/\
                                          (phase[self.aotBandIndex]*self.models[modmin].albedo[self.aotBandIndex])
                for iw in range(nwave):
                    taua[iw] = aotnirl * self.models[modmin].extc[iw] / self.models[modmin].extc[self.aotBandIndex]

                tsol = self.model_trans(im=modmin,wave=wave,nwave=nwave,theta=solz,taua=taua)
                tsen = self.model_trans(im=modmin,wave=wave,nwave=nwave,theta=senz,taua=taua)
            else:
                tauamin = np.zeros(nwave)
                tauamax = np.zeros(nwave)
                #get SS aerosol reflectance at longest sensor wavelength */
                rhoas = self.rhoa2rhos(modnum=modmin,solz=solz,senz=senz,phi=phi,wv=wv,rhoa=rhoa,wave=wave,nwave=nwave)
                phase  = self.get_phase(im=modmin,solz=solz,senz=senz,phi=phi)
                #get aerosol optical thickness at longest sensor wavelength */
                aotnirl = rhoas[self.aotBandIndex]*(4.0*csolz*csenz)/\
                                          (phase[self.aotBandIndex]*self.models[modmin].albedo[self.aotBandIndex])
                for iw in range(nwave):
                    tauamin[iw] = aotnirl * self.models[modmin].extc[iw] / self.models[modmin].extc[self.aotBandIndex]

                rhoas = self.rhoa2rhos(modnum=modmax,solz=solz,senz=senz,phi=phi,wv=wv,rhoa=rhoa,wave=wave,nwave=nwave)
                phase  = self.get_phase(im=modmax,solz=solz,senz=senz,phi=phi)
                #get aerosol optical thickness at longest sensor wavelength */
                aotnirl = rhoas[self.aotBandIndex]*(4.0*csolz*csenz)/\
                                          (phase[self.aotBandIndex]*self.models[modmax].albedo[self.aotBandIndex])
                for iw in range(nwave):
                    tauamax[iw] = aotnirl * self.models[modmax].extc[iw] / self.models[modmax].extc[self.aotBandIndex]

                tsolmin = self.model_trans(im=modmin,wave=wave,nwave=nwave,theta=solz,taua=tauamin)
                tsenmin = self.model_trans(im=modmin,wave=wave,nwave=nwave,theta=senz,taua=tauamin)
                tsolmax = self.model_trans(im=modmax,wave=wave,nwave=nwave,theta=solz,taua=tauamax)
                tsenmax = self.model_trans(im=modmax,wave=wave,nwave=nwave,theta=senz,taua=tauamax)

                tsol = (1-modrat)*tsolmin + modrat*tsolmax
                tsen = (1-modrat)*tsenmin + modrat*tsenmax

                 #correct for pressure difference from standard pressure */
                tsol = tsol*np.exp(-0.5*taur/mu0*(pr/cns.STDPR-1))
                tsen = tsen*np.exp(-0.5*taur/mu *(pr/cns.STDPR-1))

        else:
            modmin1,modmax1,modrat1 = self.getmodel_num(rhindex=mindex1,fmf=fmf)
            modmin2,modmax2,modrat2 = self.getmodel_num(rhindex=mindex2,fmf=fmf)
            if modrat1==0 and modrat2==0:

                taua1 = np.zeros(nwave)
                taua2 = np.zeros(nwave)
                #get SS aerosol reflectance at longest sensor wavelength */
                rhoas = self.rhoa2rhos(modnum=modmin1,solz=solz,senz=senz,phi=phi,wv=wv,rhoa=rhoa,wave=wave,nwave=nwave)
                phase  = self.get_phase(im=modmin1,solz=solz,senz=senz,phi=phi)
                #get aerosol optical thickness at longest sensor wavelength */
                aotnirl = rhoas[self.aotBandIndex]*(4.0*csolz*csenz)/\
                                          (phase[self.aotBandIndex]*self.models[modmin1].albedo[self.aotBandIndex])
                for iw in range(nwave):
                    taua1[iw] = aotnirl * self.models[modmin1].extc[iw] / self.models[modmin1].extc[self.aotBandIndex]

                rhoas = self.rhoa2rhos(modnum=modmin2,solz=solz,senz=senz,phi=phi,wv=wv,rhoa=rhoa,wave=wave,nwave=nwave)
                phase  = self.get_phase(im=modmin2,solz=solz,senz=senz,phi=phi)
                #get aerosol optical thickness at longest sensor wavelength */
                aotnirl = rhoas[self.aotBandIndex]*(4.0*csolz*csenz)/\
                                          (phase[self.aotBandIndex]*self.models[modmin2].albedo[self.aotBandIndex])
                for iw in range(nwave):
                    taua2[iw] = aotnirl * self.models[modmin2].extc[iw] / self.models[modmin2].extc[self.aotBandIndex]

                tsol1 = self.model_trans(im=modmin1,wave=wave,nwave=nwave,theta=solz,taua=taua1)
                tsen1 = self.model_trans(im=modmin1,wave=wave,nwave=nwave,theta=senz,taua=taua1)
                tsol2 = self.model_trans(im=modmin2,wave=wave,nwave=nwave,theta=solz,taua=taua2)
                tsen2 = self.model_trans(im=modmin2,wave=wave,nwave=nwave,theta=senz,taua=taua2)

                tsol1 = tsol1*np.exp(-0.5*taur/mu0*(pr/cns.STDPR-1))
                tsen1 = tsen1*np.exp(-0.5*taur/mu *(pr/cns.STDPR-1))

                tsol2 = tsol2*np.exp(-0.5*taur/mu0*(pr/cns.STDPR-1))
                tsen2 = tsen2*np.exp(-0.5*taur/mu *(pr/cns.STDPR-1))


                tsol = (1-wt)*tsol1 + wt*tsol2
                tsen = (1-wt)*tsen1 + wt*tsen2

            elif modrat1==0 and modrat2!=0:
                taua1 = np.zeros(nwave)
                taua2min = np.zeros(nwave)
                taua2max = np.zeros(nwave)
                #get SS aerosol reflectance at longest sensor wavelength */
                rhoas = self.rhoa2rhos(modnum=modmin1,solz=solz,senz=senz,phi=phi,wv=wv,rhoa=rhoa,wave=wave,nwave=nwave)
                phase  = self.get_phase(im=modmin1,solz=solz,senz=senz,phi=phi)
                #get aerosol optical thickness at longest sensor wavelength */
                aotnirl = rhoas[self.aotBandIndex]*(4.0*csolz*csenz)/\
                                          (phase[self.aotBandIndex]*self.models[modmin1].albedo[self.aotBandIndex])
                for iw in range(nwave):
                    taua1[iw] = aotnirl * self.models[modmin1].extc[iw] / self.models[modmin1].extc[self.aotBandIndex]

                rhoas = self.rhoa2rhos(modnum=modmin2,solz=solz,senz=senz,phi=phi,wv=wv,rhoa=rhoa,wave=wave,nwave=nwave)
                phase  = self.get_phase(im=modmin2,solz=solz,senz=senz,phi=phi)
                #get aerosol optical thickness at longest sensor wavelength */
                aotnirl = rhoas[self.aotBandIndex]*(4.0*csolz*csenz)/\
                                          (phase[self.aotBandIndex]*self.models[modmin2].albedo[self.aotBandIndex])
                for iw in range(nwave):
                    taua2min[iw] = aotnirl * self.models[modmin2].extc[iw] / self.models[modmin2].extc[self.aotBandIndex]

                rhoas = self.rhoa2rhos(modnum=modmax2,solz=solz,senz=senz,phi=phi,wv=wv,rhoa=rhoa,wave=wave,nwave=nwave)
                phase  = self.get_phase(im=modmax2,solz=solz,senz=senz,phi=phi)
                #get aerosol optical thickness at longest sensor wavelength */
                aotnirl = rhoas[self.aotBandIndex]*(4.0*csolz*csenz)/\
                                          (phase[self.aotBandIndex]*self.models[modmax2].albedo[self.aotBandIndex])
                for iw in range(nwave):
                    taua2max[iw] = aotnirl * self.models[modmax2].extc[iw] / self.models[modmax2].extc[self.aotBandIndex]


                tsol1 = self.model_trans(im=modmin1,wave=wave,nwave=nwave,theta=solz,taua=taua1)
                tsen1 = self.model_trans(im=modmin1,wave=wave,nwave=nwave,theta=senz,taua=taua1)

                tsol2min = self.model_trans(im=modmin2,wave=wave,nwave=nwave,theta=solz,taua=taua2min)
                tsen2min = self.model_trans(im=modmin2,wave=wave,nwave=nwave,theta=senz,taua=taua2min)
                tsol2max = self.model_trans(im=modmax2,wave=wave,nwave=nwave,theta=solz,taua=taua2max)
                tsen2max = self.model_trans(im=modmax2,wave=wave,nwave=nwave,theta=senz,taua=taua2max)

                tsol2 = (1-modrat2)*tsol2min + modrat2*tsol2max
                tsen2 = (1-modrat2)*tsen2min + modrat2*tsen2max

                tsol1 = tsol1*np.exp(-0.5*taur/mu0*(pr/cns.STDPR-1))
                tsen1 = tsen1*np.exp(-0.5*taur/mu *(pr/cns.STDPR-1))

                tsol2 = tsol2*np.exp(-0.5*taur/mu0*(pr/cns.STDPR-1))
                tsen2 = tsen2*np.exp(-0.5*taur/mu *(pr/cns.STDPR-1))

                tsol = (1-wt)*tsol1 + wt*tsol2
                tsen = (1-wt)*tsen1 + wt*tsen2

            elif modrat1!=0 and modrat2==0:
                taua2 = np.zeros(nwave)
                taua1min = np.zeros(nwave)
                taua1max = np.zeros(nwave)
                #get SS aerosol reflectance at longest sensor wavelength */
                rhoas = self.rhoa2rhos(modnum=modmin2,solz=solz,senz=senz,phi=phi,wv=wv,rhoa=rhoa,wave=wave,nwave=nwave)
                phase  = self.get_phase(im=modmin2,solz=solz,senz=senz,phi=phi)
                #get aerosol optical thickness at longest sensor wavelength */
                aotnirl = rhoas[self.aotBandIndex]*(4.0*csolz*csenz)/\
                                          (phase[self.aotBandIndex]*self.models[modmin2].albedo[self.aotBandIndex])
                for iw in range(nwave):
                    taua2[iw] = aotnirl * self.models[modmin2].extc[iw] / self.models[modmin2].extc[self.aotBandIndex]

                rhoas = self.rhoa2rhos(modnum=modmin1,solz=solz,senz=senz,phi=phi,wv=wv,rhoa=rhoa,wave=wave,nwave=nwave)
                phase  = self.get_phase(im=modmin1,solz=solz,senz=senz,phi=phi)
                #get aerosol optical thickness at longest sensor wavelength */
                aotnirl = rhoas[self.aotBandIndex]*(4.0*csolz*csenz)/\
                                          (phase[self.aotBandIndex]*self.models[modmin1].albedo[self.aotBandIndex])
                for iw in range(nwave):
                    taua1min[iw] = aotnirl * self.models[modmin1].extc[iw] / self.models[modmin1].extc[self.aotBandIndex]

                rhoas = self.rhoa2rhos(modnum=modmax1,solz=solz,senz=senz,phi=phi,wv=wv,rhoa=rhoa,wave=wave,nwave=nwave)
                phase  = self.get_phase(im=modmax1,solz=solz,senz=senz,phi=phi)
                #get aerosol optical thickness at longest sensor wavelength */
                aotnirl = rhoas[self.aotBandIndex]*(4.0*csolz*csenz)/\
                                          (phase[self.aotBandIndex]*self.models[modmax1].albedo[self.aotBandIndex])
                for iw in range(nwave):
                    taua1max[iw] = aotnirl * self.models[modmax1].extc[iw] / self.models[modmax1].extc[self.aotBandIndex]

                tsol2 = self.model_trans(im=modmin2,wave=wave,nwave=nwave,theta=solz,taua=taua2)
                tsen2 = self.model_trans(im=modmin2,wave=wave,nwave=nwave,theta=senz,taua=taua2)

                tsol1min = self.model_trans(im=modmin1,wave=wave,nwave=nwave,theta=solz,taua=taua1min)
                tsen1min = self.model_trans(im=modmin1,wave=wave,nwave=nwave,theta=senz,taua=taua1min)
                tsol1max = self.model_trans(im=modmax1,wave=wave,nwave=nwave,theta=solz,taua=taua1max)
                tsen1max = self.model_trans(im=modmax1,wave=wave,nwave=nwave,theta=senz,taua=taua1max)

                tsol1 = (1-modrat2)*tsol1min + modrat2*tsol1max
                tsen1 = (1-modrat2)*tsen1min + modrat2*tsen1max

                tsol1 = tsol1*np.exp(-0.5*taur/mu0*(pr/cns.STDPR-1))
                tsen1 = tsen1*np.exp(-0.5*taur/mu *(pr/cns.STDPR-1))

                tsol2 = tsol2*np.exp(-0.5*taur/mu0*(pr/cns.STDPR-1))
                tsen2 = tsen2*np.exp(-0.5*taur/mu *(pr/cns.STDPR-1))

                tsol = (1-wt)*tsol1 + wt*tsol2
                tsen = (1-wt)*tsen1 + wt*tsen2
            else:
                taua1min = np.zeros(nwave)
                taua1max = np.zeros(nwave)
                taua2min = np.zeros(nwave)
                taua2max = np.zeros(nwave)

                rhoas = self.rhoa2rhos(modnum=modmin1,solz=solz,senz=senz,phi=phi,wv=wv,rhoa=rhoa,wave=wave,nwave=nwave)
                phase  = self.get_phase(im=modmin1,solz=solz,senz=senz,phi=phi)
                #get aerosol optical thickness at longest sensor wavelength */
                aotnirl = rhoas[self.aotBandIndex]*(4.0*csolz*csenz)/\
                                          (phase[self.aotBandIndex]*self.models[modmin1].albedo[self.aotBandIndex])
                for iw in range(nwave):
                    taua1min[iw] = aotnirl * self.models[modmin1].extc[iw] / self.models[modmin1].extc[self.aotBandIndex]

                rhoas = self.rhoa2rhos(modnum=modmax1,solz=solz,senz=senz,phi=phi,wv=wv,rhoa=rhoa,wave=wave,nwave=nwave)
                phase  = self.get_phase(im=modmax1,solz=solz,senz=senz,phi=phi)
                #get aerosol optical thickness at longest sensor wavelength */
                aotnirl = rhoas[self.aotBandIndex]*(4.0*csolz*csenz)/\
                                          (phase[self.aotBandIndex]*self.models[modmax1].albedo[self.aotBandIndex])
                for iw in range(nwave):
                    taua1max[iw] = aotnirl * self.models[modmax1].extc[iw] / self.models[modmax1].extc[self.aotBandIndex]

                rhoas = self.rhoa2rhos(modnum=modmin2,solz=solz,senz=senz,phi=phi,wv=wv,rhoa=rhoa,wave=wave,nwave=nwave)
                phase  = self.get_phase(im=modmin2,solz=solz,senz=senz,phi=phi)
                #get aerosol optical thickness at longest sensor wavelength */
                aotnirl = rhoas[self.aotBandIndex]*(4.0*csolz*csenz)/\
                                          (phase[self.aotBandIndex]*self.models[modmin2].albedo[self.aotBandIndex])
                for iw in range(nwave):
                    taua2min[iw] = aotnirl * self.models[modmin2].extc[iw] / self.models[modmin2].extc[self.aotBandIndex]

                rhoas = self.rhoa2rhos(modnum=modmax2,solz=solz,senz=senz,phi=phi,wv=wv,rhoa=rhoa,wave=wave,nwave=nwave)
                phase  = self.get_phase(im=modmax2,solz=solz,senz=senz,phi=phi)
                #get aerosol optical thickness at longest sensor wavelength */
                aotnirl = rhoas[self.aotBandIndex]*(4.0*csolz*csenz)/\
                                          (phase[self.aotBandIndex]*self.models[modmax2].albedo[self.aotBandIndex])
                for iw in range(nwave):
                    taua2max[iw] = aotnirl * self.models[modmax2].extc[iw] / self.models[modmax2].extc[self.aotBandIndex]

                tsol1min = self.model_trans(im=modmin1,wave=wave,nwave=nwave,theta=solz,taua=taua1min)
                tsen1min = self.model_trans(im=modmin1,wave=wave,nwave=nwave,theta=senz,taua=taua1min)
                tsol1max = self.model_trans(im=modmax1,wave=wave,nwave=nwave,theta=solz,taua=taua1max)
                tsen1max = self.model_trans(im=modmax1,wave=wave,nwave=nwave,theta=senz,taua=taua1max)

                tsol2min = self.model_trans(im=modmin2,wave=wave,nwave=nwave,theta=solz,taua=taua2min)
                tsen2min = self.model_trans(im=modmin2,wave=wave,nwave=nwave,theta=senz,taua=taua2min)
                tsol2max = self.model_trans(im=modmax2,wave=wave,nwave=nwave,theta=solz,taua=taua2max)
                tsen2max = self.model_trans(im=modmax2,wave=wave,nwave=nwave,theta=senz,taua=taua2max)

                tsol1 = (1-modrat2)*tsol1min + modrat2*tsol1max
                tsen1 = (1-modrat2)*tsen1min + modrat2*tsen1max
                tsol2 = (1-modrat2)*tsol2min + modrat2*tsol2max
                tsen2 = (1-modrat2)*tsen2min + modrat2*tsen2max

                tsol1 = tsol1*np.exp(-0.5*taur/mu0*(pr/cns.STDPR-1))
                tsen1 = tsen1*np.exp(-0.5*taur/mu *(pr/cns.STDPR-1))

                tsol2 = tsol2*np.exp(-0.5*taur/mu0*(pr/cns.STDPR-1))
                tsen2 = tsen2*np.exp(-0.5*taur/mu *(pr/cns.STDPR-1))

                tsol = (1-wt)*tsol1 + wt*tsol2
                tsen = (1-wt)*tsen1 + wt*tsen2



        return (tsol,tsen)

    #get diffuse transmittance for a specific aerosol model
    #this is for a single aerosol model
    def model_trans(self,im,wave,nwave,theta,taua):
        dtran = np.zeros(nwave)
        if self.firstCall_transmittance==1:
            self.firstCall_transmittance = 0
            for iw in range(nwave):
                if abs(self.wave[iw] - self.dtran_wave[iw]) > 0.51:
                    um1 = self.dtran_wave[iw]/1000.0
                    um2 = self.wave[iw]/1000.0
                    taur1 = 0.008569*np.pow(um1,-4)*(1.0+(0.0113*np.pow(um1,-2))+(0.00013*np.pow(um1,-4)))
                    taur2 = 0.008569*np.pow(um2,-4)*(1.0+(0.0113*np.pow(um2,-2))+(0.00013*np.pow(um2,-4)))
                    self.intexp[iw] = taur2/taur1
                    self.inttst[iw] = 1
                    print("Interpolating diffuse transmittance for %d from %f by %f\n"%self.wave[iw],self.dtran_wave[iw],self.intexp[iw])

        #find bracketing zenith angle indices */
        for i in range(self.dtran_ntheta):
            if theta < self.dtran_theta[i]:
                break

        if i == self.dtran_ntheta:
            i1 = i-1
            i2 = i-1
            wt = 0.0
        else:
            i1 =min(max(i-1,0),self.dtran_ntheta-2)
            i2 = i1+1
            x1   = self.dtran_airmass[i1]
            x2   = self.dtran_airmass[i2]
            xbar = 1.0/np.cos(theta/cns.RADEG)
            wt   = (xbar-x1)/(x2-x1)

        #use coefficients of nearest wavelength */
        for iw in range(nwave):
            #iwtab = iwdtab[iw];
            a1  = self.models[im].dtran_a[iw][i1]
            b1  = self.models[im].dtran_b[iw][i1]
            a2  = self.models[im].dtran_a[iw][i2]
            b2  = self.models[im].dtran_b[iw][i2]

            if self.inttst[iw]==1:
                a1 = np.pow(a1,self.intexp[iw])
                a2 = np.pow(a2,self.intexp[iw])


            y1 = a1 * np.exp(-b1*taua[iw])
            y2 = a2 * np.exp(-b2*taua[iw])

            dtran[iw] = max(min( (1.0-wt)*y1 + wt*y2, 1.0), 1e-5)

        return dtran

    #get rhoa for specific rh,fmf and aot(nirl)
    #nwave refers to the first nwave wavelength will be calculated
    def get_rhoa_fixedaot(self,rh,fmf,aotnirl,solz,senz,phi,wv,nwave):
        mindex1,mindex2,wt = self.getrh_index(rh=rh)
        rhoa = None
        aot = None
        if wt==0:
            modmin,modmax,modrat = self.getmodel_num(rhindex=mindex1,fmf=fmf)
            print(modmin,modmax)
            if modrat==0:
                rhoa,aot = self.comp_rhoa_fixedaot(im=modmin,aotnirl=aotnirl,solz=solz,senz=senz,phi=phi,wv=wv,nwave=nwave)
            else:
                rhoamin,aotmin = self.comp_rhoa_fixedaot(im=modmin,aotnirl=aotnirl,solz=solz,senz=senz,phi=phi,wv=wv,nwave=nwave)
                rhoamax,aotmax = self.comp_rhoa_fixedaot(im=modmax,aotnirl=aotnirl,solz=solz,senz=senz,phi=phi,wv=wv,nwave=nwave)
                rhoa = (1-modrat)*rhoamin + modrat*rhoamax
                aot = (1-modrat)*aotmin + modrat*aotmax
        else:
            modmin1,modmax1,modrat1 = self.getmodel_num(rhindex=mindex1,fmf=fmf)
            modmin2,modmax2,modrat2 = self.getmodel_num(rhindex=mindex2,fmf=fmf)

            if modrat1==0 and modrat2==0:
                rhoa1,aot1 = self.comp_rhoa_fixedaot(im=modmin1,aotnirl=aotnirl,solz=solz,senz=senz,phi=phi,wv=wv,nwave=nwave)
                rhoa2,aot2 = self.comp_rhoa_fixedaot(im=modmin2,aotnirl=aotnirl,solz=solz,senz=senz,phi=phi,wv=wv,nwave=nwave)
                rhoa = (1-wt)*rhoa1 + wt*rhoa2
                aot = (1-wt)*aot1 + wt*aot2

            elif modrat1==0 and modrat2!=0:
                rhoa1,aot1 = self.comp_rhoa_fixedaot(im=modmin1,aotnirl=aotnirl,solz=solz,senz=senz,phi=phi,wv=wv,nwave=nwave)
                rhoa2min,aot2min = self.comp_rhoa_fixedaot(im=modmin2,aotnirl=aotnirl,solz=solz,senz=senz,phi=phi,wv=wv,nwave=nwave)
                rhoa2max,aot2max = self.comp_rhoa_fixedaot(im=modmax2,aotnirl=aotnirl,solz=solz,senz=senz,phi=phi,wv=wv,nwave=nwave)
                rhoa2 = (1-modrat2)*rhoa2min + modrat2*rhoa2max
                aot2 = (1-modrat2)*aot2min + modrat2*aot2max
                aot = (1-wt)*aot1 + wt*aot2
                rhoa = (1-wt)*rhoa1 + wt*rhoa2

            elif modrat1!=0 and modrat2==0:
                rhoa2,aot2 = self.comp_rhoa_fixedaot(im=modmin2,aotnirl=aotnirl,solz=solz,senz=senz,phi=phi,wv=wv,nwave=nwave)
                rhoa1min,aot1min = self.comp_rhoa_fixedaot(im=modmin1,aotnirl=aotnirl,solz=solz,senz=senz,phi=phi,wv=wv,nwave=nwave)
                rhoa1max,aot1max = self.comp_rhoa_fixedaot(im=modmax1,aotnirl=aotnirl,solz=solz,senz=senz,phi=phi,wv=wv,nwave=nwave)
                rhoa1 = (1-modrat1)*rhoa1min + modrat1*rhoa1max
                aot1 = (1-modrat1)*aot1min + modrat1*aot1max
                aot = (1-wt)*aot1 + wt*aot2
                rhoa = (1-wt)*rhoa1 + wt*rhoa2
            else:
                rhoa1min,aot1min = self.comp_rhoa_fixedaot(im=modmin1,aotnirl=aotnirl,solz=solz,senz=senz,phi=phi,wv=wv,nwave=nwave)
                rhoa1max,aot1max = self.comp_rhoa_fixedaot(im=modmax1,aotnirl=aotnirl,solz=solz,senz=senz,phi=phi,wv=wv,nwave=nwave)
                rhoa2min,aot2min = self.comp_rhoa_fixedaot(im=modmin2,aotnirl=aotnirl,solz=solz,senz=senz,phi=phi,wv=wv,nwave=nwave)
                rhoa2max,aot2max = self.comp_rhoa_fixedaot(im=modmax2,aotnirl=aotnirl,solz=solz,senz=senz,phi=phi,wv=wv,nwave=nwave)
                rhoa2 = (1-modrat2)*rhoa2min + modrat2*rhoa2max
                rhoa1 = (1-modrat1)*rhoa1min + modrat1*rhoa1max
                aot1 = (1-modrat1)*aot1min + modrat1*aot1max
                aot2 = (1-modrat2)*aot2min + modrat2*aot2max

                aot = (1-wt)*aot1 + wt*aot2
                rhoa = (1-wt)*rhoa1 + wt*rhoa2

        return (rhoa,aot)


    #get rhoa and taua by fixed AOT, fixedaot()
    # this is for a single aerosol model
    #return rhoa and aot for given bands,nwave means the first nwave bands will be calculated
    def comp_rhoa_fixedaot(self,im,aotnirl,solz,senz,phi,wv,nwave):
        #nwave = self.nwave
        f = np.zeros(nwave)
        aot = np.zeros(nwave)
        lnf1 = np.zeros(nwave)
        rhoas = np.zeros(nwave)
        rhoa = np.zeros(nwave)
        mu0 = np.cos(solz/cns.RADEG)
        mu = np.cos(senz/cns.RADEG)
        airmass = 1.0/mu0 + 1.0/mu
        tempModel = self.models[im]

        # get aerosol optical thickness at all other table wavelengths */
        for iw in range(nwave):
            aot[iw] = aotnirl * tempModel.extc[iw] / tempModel.extc[self.aotBandIndex]

        for iw in range(nwave):
            if aot[iw]<=0:
                return -1
        phase = self.get_phase(im,solz,senz,phi)

        #compute factor for SS approximation, set-up for interpolation
        for iw in range(nwave):
            f[iw] = tempModel.albedo[iw]*phase[iw]/4.0/mu0/mu
            if tempModel.interpNeeded:
                lnf1[iw] = np.log(f[iw])

        for iw in range(nwave):
            rhoas[iw] = aot[iw]*f[iw]

        ac,bc,cc = self.ss2ms_coef(im,solz,senz,phi)
        print(ac)
        print(bc)
        print(cc)
        cf = self.aeroob_cf(im,solz,senz,phi)

        for iw in range(nwave):
            if rhoas[iw] < 1.e-20:
                rhoa[iw] = rhoas[iw]
            else:
                a = ac[iw]
                b = bc[iw]
                c = cc[iw]
                lnrhoas = np.log(rhoas[iw]*self.si.aeroob(iw=iw,cf=cf,wv=wv,airmass=airmass))
                rhoa[iw] = np.exp(a + b*lnrhoas + c*lnrhoas*lnrhoas)

        return (rhoa,aot)

    #get model phase for a specific observe geometry
    #get phase
    def get_phase(self,im,solz,senz,phi):
        phase = []
        tempModel = self.models[im]
        if tempModel.phase_geometry.has_key((solz,senz,phi)):
            return tempModel.phase_geometry[(solz,senz,phi)]
        else:
            csolz = np.cos(solz/cns.RADEG)
            csenz = np.cos(senz/cns.RADEG)
            cphi  = np.cos(phi /cns.RADEG)
            temp   = np.sqrt((1.0-csenz*csenz)*(1.0-csolz*csolz)) * cphi
            scatt1 = np.arccos(max(-csenz*csolz + temp,-1.0))*cns.RADEG
            scatt2 = np.arccos(min( csenz*csolz + temp, 1.0))*cns.RADEG

            #compute Fresnel coefficients
            fres1 = ult.fresnel_coef(csenz,self.nw)
            fres2 = ult.fresnel_coef(csolz,self.nw)

            for iw in range(self.nwave):
                phase1 = interpolate.splev(scatt1, tempModel.tck[iw])
                phase2 = interpolate.splev(scatt2, tempModel.tck[iw])
                phasetemp = np.exp(phase1) + np.exp(phase2)*(fres1+fres2)
                phase.append(phasetemp)

        tempModel.phase_geometry[(solz,senz,phi)] = phase
        return phase


    def INDEX(self,iw,isol,iphi,isen):
        return iw*self.nsolz*self.nphi*self.nsenz +\
               isol*self.nphi*self.nsenz + iphi*self.nsenz + isen

    #single scattering to multiscattering
    def rhos2rhoa(self,sensorID,solz,senz,phi,wv,rhos,wave,nwave):
        return 0

    # multiscattering scattering to single
    def rhoa2rhos(self,modnum,solz,senz,phi,wv,rhoa,wave,nwave):
        status = 0
        mu0 = np.cos(solz/cns.RADEG)
        mu = np.cos(senz/cns.RADEG)
        airmass = 1.0/mu0 + 1.0/mu
        rhoas = np.zeros(nwave)
        ac,bc,cc=self.ss2ms_coef(im=modnum,solz=solz,senz=senz,phi=phi)
        cf = self.aeroob_cf(modnum,solz,senz,phi)
        for iw in range(nwave):
            if rhoa[iw] < 1.e-20:
                rhoas[iw] = rhoa[iw]
            else:
                a = ac[iw]
                b = bc[iw]
                c = cc[iw]
                f = b*b - 4*c*( a - np.log(rhoa[iw]))
            if f > 0.00001:
                if abs(c) > 1.e-20:
                    rhoas[iw] = np.exp(0.5*(-b+np.sqrt(f))/c)
            elif abs(a) > 1.e-20 and abs(b) > 1.e-20:
                rhoas[iw] = pow(rhoa[iw]/a, 1./b)
            else:
                status = 1
                break

            rhoas[iw] = rhoas[iw]/self.si.aeroob(iw,cf,wv,airmass)

        return rhoas

    #aeroob_cf() - out-of-band water-vapor scale factor
    def aeroob_cf(self,modnum,solz,senz,phi):
        if self.firstCall_aeroob_cf:
            iw1 = ult.windex(765,self.wave,self.nwave)
            iw2 = ult.windex(865,self.wave,self.nwave)
            if iw1 == iw2:
                iw1 = iw1-1
            self.iw1_aeroob_cf = iw1
            self.iw2_aeroob_cf = iw2
            self.firstCall_aeroob_cf = 0

        phase  = self.get_phase(modnum,solz,senz,phi)
        rhoas1 = self.models[modnum].albedo[self.iw1_aeroob_cf] * phase[self.iw1_aeroob_cf] * self.models[modnum].extc[self.iw1_aeroob_cf]
        rhoas2 = self.models[modnum].albedo[self.iw2_aeroob_cf] * phase[self.iw2_aeroob_cf] * self.models[modnum].extc[self.iw2_aeroob_cf]
        eps    = rhoas1/rhoas2
        cf     = np.log(eps)/(self.wave[self.iw2_aeroob_cf]-self.wave[self.iw1_aeroob_cf])

        return cf

    #load aerosol models
    def load(self,verbose=False):
        if verbose:
            print ('load aerosol models from :%s.....'%self.aerosolpath)
        for im in range(self.nmodel+1):
            if im < self.nmodel:
                aerosolfile = os.path.join(self.aerosolpath,'aerosol'+'_'+self.sensorName+'_'+self.modelnames[im]+'.hdf')
                try:
                    self.openSD=SD(aerosolfile)
                except HDF4Error, msg:
                    print ("HDF4ERROR--:%s"%msg)

                rh = float(self.modelnames[im][1:3])
                fmf = float(self.modelnames[im][4:6])
                if self.modelDic.has_key(rh) == False:
                    self.modelDic[rh] = [fmf]
                else:
                    self.modelDic[rh].append(fmf)
                if verbose:
                    print('loading '+str(im+1)+' model,rh:'+str(rh)+' fmf:'+str(fmf)+'\n')

            else:
                aerosolfile = os.path.join(self.aerosolpath,'aerosol'+'_'+self.sensorName+'_default.hdf')
                try:
                    self.openSD=SD(aerosolfile)
                except HDF4Error, msg:
                    print ("HDF4ERROR--:%s"%msg)
                    exit(-1)

            attributes=self.openSD.attributes()
            if im==0:
                self.nwave = attributes['Number of Wavelengths']

                self.intexp = np.ones(self.nwave)
                self.inttst = np.zeros(self.nwave,dtype=int)

                self.nsolz = attributes['Number of Solar Zenith Angles']
                self.nsenz = attributes['Number of View Zenith Angles']
                self.nphi = attributes['Number of Relative Azimuth Angles']
                self.nscatt = attributes['Number of Scattering Angles']
                self.dtran_nwave = attributes['Number of Diffuse Transmittance Wavelengths']
                self.dtran_ntheta = attributes['Number of Diffuse Transmittance Zenith Angles']
                self.dtran_airmass = np.zeros(self.dtran_ntheta)
                if verbose:
                    print("Number of Wavelengths                          %d\n"%self.nwave)
                    print("Number of Solar Zenith Angles                  %d\n"%self.nsolz)
                    print("Number of View Zenith Angles                   %d\n"%self.nsenz)
                    print("Number of Relative Azimuth Angles              %d\n"%self.nphi)
                    print("Number of Scattering Angles                    %d\n"%self.nscatt)
                    print("Number of Diffuse Transmittance Wavelengths    %d\n"%self.dtran_nwave)
                    print("Number of Diffuse Transmittance Zenith Angles  %d\n"%self.dtran_ntheta)

                # read wavelength
                try:
                    self.openSubSD=self.openSD.select('wave')
                    self.wave=self.openSubSD.get()
                except HDF4Error,msg:
                    print("HDF4ERROR--:%s"%msg)
                    exit(-1)
                finally:
                    self.openSubSD.endaccess()

                # read solar zenith
                try:
                    self.openSubSD=self.openSD.select('solz')
                    self.solz=self.openSubSD.get()
                except HDF4Error,msg:
                    print("HDF4ERROR--:%s"%msg)
                    exit(-1)
                finally:
                    self.openSubSD.endaccess()

                 # read senz zenith
                try:
                    self.openSubSD=self.openSD.select('senz')
                    self.senz=self.openSubSD.get()
                except HDF4Error,msg:
                    print("HDF4ERROR--:%s"%msg)
                    exit(-1)
                finally:
                    self.openSubSD.endaccess()


                 # read phi zenith
                try:
                    self.openSubSD=self.openSD.select('phi')
                    self.phi=self.openSubSD.get()
                except HDF4Error,msg:
                    print("HDF4ERROR--:%s"%msg)
                    exit(-1)
                finally:
                    self.openSubSD.endaccess()

                 # read scatt zenith
                try:
                    self.openSubSD=self.openSD.select('scatt')
                    self.scatt=self.openSubSD.get()
                except HDF4Error,msg:
                    print("HDF4ERROR--:%s"%msg)
                    exit(-1)
                finally:
                    self.openSubSD.endaccess()

                 # read dtran_wave zenith
                try:
                    self.openSubSD=self.openSD.select('dtran_wave')
                    self.dtran_wave=self.openSubSD.get()
                except HDF4Error,msg:
                    print("HDF4ERROR--:%s"%msg)
                    exit(-1)
                finally:
                    self.openSubSD.endaccess()


                # read dtran_theta zenith
                try:
                    self.openSubSD=self.openSD.select('dtran_theta')
                    self.dtran_theta=self.openSubSD.get()
                except HDF4Error,msg:
                    print("HDF4ERROR--:%s"%msg)
                    exit(-1)
                finally:
                    self.openSubSD.endaccess()




            mname=""
            if im<self.nmodel:
                mname=self.modelnames[im]
            else:
                mname='default'

            model= am.Aeromodel(name=mname,nbands=self.nwave)

            model.rh=attributes['Relative Humidity']
            model.sd=attributes['Size Distribution']

            # read albedo zenith
            try:
                self.openSubSD=self.openSD.select('albedo')
                model.albedo=self.openSubSD.get()
            except HDF4Error,msg:
                print("HDF4ERROR--:%s"%msg)
                exit(-1)
            finally:
                self.openSubSD.endaccess()

            # read extc
            try:
                self.openSubSD=self.openSD.select('extc')
                model.extc=self.openSubSD.get()
            except HDF4Error,msg:
                print("HDF4ERROR--:%s"%msg)
                exit(-1)
            finally:
                self.openSubSD.endaccess()
                
                        # read phase
            try:
                self.openSubSD=self.openSD.select('scatt')
                model.scatt=self.openSubSD.get()
            except HDF4Error,msg:
                print("HDF4ERROR--:%s"%msg)
                exit(-1)
            finally:
                self.openSubSD.endaccess()

            # read phase
            try:
                self.openSubSD=self.openSD.select('phase')
                model.phase=self.openSubSD.get()
            except HDF4Error,msg:
                print("HDF4ERROR--:%s"%msg)
                exit(-1)
            finally:
                self.openSubSD.endaccess()

            # read acost
            try:
                self.openSubSD=self.openSD.select('acost')
                model.acost=self.openSubSD.get()
                self.acostInterpolator = RegularGridInterpolator((self.wave,self.solz,self.phi,self.senz),
                                                                 model.acost)
            except HDF4Error,msg:
                print("HDF4ERROR--:%s"%msg)
                exit(-1)
            finally:
                self.openSubSD.endaccess()

            # read bcost
            try:
                self.openSubSD=self.openSD.select('bcost')
                model.bcost=self.openSubSD.get()
                self.bcostInterpolator = RegularGridInterpolator((self.wave,self.solz,self.phi,self.senz),
                                                                 model.bcost)
            except HDF4Error as msg:
                print("HDF4ERROR--:%s"%msg)
                exit(-1)
            finally:
                self.openSubSD.endaccess()

            # read ccost
            try:
                self.openSubSD=self.openSD.select('ccost')
                model.ccost=self.openSubSD.get()
                self.ccostInterpolator = RegularGridInterpolator((self.wave,self.solz,self.phi,self.senz),
                                                                 model.ccost)
            except HDF4Error as msg:
                print("HDF4ERROR--:%s"%msg)
                exit(-1)
            finally:
                self.openSubSD.endaccess()

            # read dtran_a
            try:
                self.openSubSD=self.openSD.select('dtran_a')
                model.dtran_a=self.openSubSD.get()
            except HDF4Error as msg:
                print("HDF4ERROR--:%s"%msg)
                exit(-1)
            finally:
                self.openSubSD.endaccess()

             # read dtran_b
            try:
                self.openSubSD=self.openSD.select('dtran_b')
                model.dtran_b=self.openSubSD.get()
            except HDF4Error as msg:
                print("HDF4ERROR--:%s"%msg)
                exit(-1)
            finally:
                self.openSubSD.endaccess()


            '''
                to do......
                model.angstrom

                model.lnphase
                model.d2phase
            '''

            #pre-compute model.angstrom
            iwbase = ult.windex(865, self.wave,self.nwave)
            for iw in range(self.nwave):
                if iw != iwbase:
                    model.angstrom[iw] = -np.log(model.extc[iw]/model.extc[iwbase])/np.log(self.wave[iw]/self.wave[iwbase])

            model.angstrom[iwbase] = model.angstrom[iwbase-1]

            #pre-compute log of phase function and 2rd derivative (for cubic interp)
            model.lnphase = np.zeros_like(model.phase)
            tcks = []
            for iw in range(self.nwave):
                for iss in range(self.nscatt):
                    model.lnphase[iw][iss]=np.log(model.phase[iw][iss])

                # d1phase1 = ult.first_deriv(self.scatt,model.lnphase[iw],0)
                # d1phase2 = ult.first_deriv(self.scatt,model.lnphase[iw],self.nscatt)

                #cubic spline interpolation
                tck=interpolate.splrep(self.scatt,model.lnphase[iw])
                tcks.append(tck)

            model.tck=tcks


            self.models.append(model)


        self.rhs = self.modelDic.keys()
        self.rhs.sort()
        self.fmfs = self.modelDic.values()[0]
        self.fmfs.sort(reverse=True)
        # precompute airmass for diffuse transmittance */

        for iss in range(self.dtran_ntheta):
            self.dtran_airmass[iss] = 1.0/np.cos(self.dtran_theta[iss]/cns.RADEG)
        if verbose:
            print ('loading aerosol models completed\n')

        return 0

    # get min and model index for a specific rh
    def getrh_index(self,rh):

        if rh < self.rhs[0]:
            rh = self.rhs[0]
        if rh > self.rhs[-1]:
            rh = self.rhs[-1]

        rhnummin = 0
        rhmunmax = 0
        rhmin = 0
        rhmax = 0
        #index rh
        for i in range(len(self.rhs)-1):
            if rh == self.rhs[i]:
                rhnummin = i*10
                rhmunmax = rhnummin
                rhmin = self.rhs[i]
                rhmax = rhmin
                break
            if rh > self.rhs[i] and rh<self.rhs[i+1]:
                rhnummin = i*10
                rhmunmax = (i+1)*10
                rhmin = self.rhs[i]
                rhmax = self.rhs[i+1]
                break

        if i == len(self.rhs)-2:
            rhnummin = (i+1)*10
            rhmunmax =rhnummin
            rhmin = self.rhs[i+1]
            rhmax = rhmin

        rhindex1 = range(rhnummin,rhnummin+10)
        rhindex2 = range(rhmunmax,rhmunmax+10)
        if rhmax==rhmin:
            wt = 0
        else:
            wt = (rh - rhmin)/(rhmax-rhmin)

        self.mindex1 = rhindex1
        self.mindex2 = rhindex2
        self.wt = wt
        return (rhindex1,rhindex2,wt)

    # get min and max model number, modrat
    def getmodel_num(self,rhindex,fmf):
        modmin = 0
        modmax = 0
        modrat = 0

        fmfnummin = 0
        fmfnummax = 0
        fmfmin = 0
        fmfmax = 0

        #index fmf
        for i in range(len(self.fmfs)-1):
            if fmf == self.fmfs[i]:
                fmfnummin = i
                fmfnummax = fmfnummin
                break
            elif fmf < self.fmfs[i] and fmf>self.fmfs[i+1]:
                fmfnummax = i
                fmfnummin = i+1
                break

        if i == len(self.fmfs)-2:
            fmfnummax = (i+1)
            fmfnummin = fmfnummax

        fmfmin = self.fmfs[fmfnummin]
        fmfmax = self.fmfs[fmfnummax]


        modmin = rhindex[fmfnummin]
        modmax = rhindex[fmfnummax]

        if fmfmin != fmfmax:
            modrat = (fmf - fmfmin)/(fmfmax-fmfmin)

        return (modmin,modmax,modrat)








