'''
@auther Pan Yanqun
@2016.3.18
'''

import os,sys
import numpy as np
from scipy import interpolate


class Aeromodel:

    def __init__(self,name,nbands):

        self.name = ''
        self.interpNeeded = 0 #is interpolation needed
        self.rh = 0.0
        self.sd = 0

         # angstrom exponent (nbands+1)*/
        self.angstrom=np.zeros(nbands+1)
         # single-scattering albedo(nbands+1), extinction coefficient(nbands+1), phase function */
        self.albedo=None
        self.extc=None
        self.phase=None
        self.scatt=None
        # /* quadratic coefficients for SS to MS relationship */
        self.acost=None
        self.bcost=None
        self.ccost=None
        # /* cubic coefficients for ms_epsilon atmospheric correction ..ZA */
        self.ams_all=None
        self.bms_all=None
        self.cms_all=None
        self.dms_all=None
        self.ems_all=None

        # /* Rayleigh-aerosol diffuse transmittance coeffs */
        self.dtran_a=None
        self.dtran_b=None

        # /* derived quantities */
        self.lnphase=None
        self.d2phase=None

        #cubic spline interpolation
        self.tck=None

        self.lastsolz = -999.
        self.lastsenz = -999.
        self.lastphi  = -999.

        self.nwave = nbands

        #save phase of observe geometry
        #format is {(solz,senl,phi):''}
        self.phase_geometry = {}
        self.ss2ms_parameters = {}
