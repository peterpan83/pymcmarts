'''
@auther Pan Yanqun
@2016.3.21
'''

import numpy as np


AEROSOLNAMES=["r30f95v01", "r30f80v01", "r30f50v01", "r30f30v01", "r30f20v01", "r30f10v01", "r30f05v01", "r30f02v01", "r30f01v01", "r30f00v01",
                "r50f95v01", "r50f80v01", "r50f50v01", "r50f30v01", "r50f20v01", "r50f10v01", "r50f05v01", "r50f02v01", "r50f01v01", "r50f00v01",
                "r70f95v01", "r70f80v01", "r70f50v01", "r70f30v01", "r70f20v01", "r70f10v01", "r70f05v01", "r70f02v01", "r70f01v01", "r70f00v01",
                "r75f95v01", "r75f80v01", "r75f50v01", "r75f30v01", "r75f20v01", "r75f10v01", "r75f05v01", "r75f02v01", "r75f01v01", "r75f00v01",
                "r80f95v01", "r80f80v01", "r80f50v01", "r80f30v01", "r80f20v01", "r80f10v01", "r80f05v01", "r80f02v01", "r80f01v01", "r80f00v01",
                "r85f95v01", "r85f80v01", "r85f50v01", "r85f30v01", "r85f20v01", "r85f10v01", "r85f05v01", "r85f02v01", "r85f01v01", "r85f00v01",
                "r90f95v01", "r90f80v01", "r90f50v01", "r90f30v01", "r90f20v01", "r90f10v01", "r90f05v01", "r90f02v01", "r90f01v01", "r90f00v01",
                "r95f95v01", "r95f80v01", "r95f50v01", "r95f30v01", "r95f20v01", "r95f10v01", "r95f05v01", "r95f02v01", "r95f01v01", "r95f00v01"]

OCSSWDATA = '/home/pan/ocssw/share'

#return wavelength index of table which is closest to sensor wavelength
#wave=float,twave=[],ntwave=int
def windex(wave,twave,ntwave):
    index=-1
    wdiff=0
    wdiffmin=99999.
    for i in range(ntwave):
        if twave[i]==wave:
            index=i

        wdiff = abs(twave[i]-wave)
        if wdiff < wdiffmin:
            wdiffmin = wdiff
            index=i

    return index

#return first derivative(dy/dx) of 1st and last array indices using a 4-pt Lagrangian interpolation,
#Notice:it is assume that 4 points exist
def first_deriv(x, y, n):
    a1 = x[0]-x[1]
    a2 = x[0]-x[2]
    a3 = x[0]-x[3]
    a4 = x[1]-x[2]
    a5 = x[1]-x[3]
    a6 = x[2]-x[3]
    d1 = 0.
    if (n == 0):
        d1 = y[0]*(1.0/a1+1.0/a2+1.0/a3)- a2*a3*y[1]/(a1*a4*a5)+ a1*a3*y[2]/(a2*a4*a6)- a1*a2*y[3]/(a3*a5*a6)

    else:
        a1 = x[n-1]-x[n-4]
        a2 = x[n-1]-x[n-3]
        a3 = x[n-1]-x[n-2]
        a4 = x[n-2]-x[n-4]
        a5 = x[n-2]-x[n-3]
        a6 = x[n-3]-x[n-4]
        d1 = -a2*a3*y[n-4]/(a6*a4*a1)+  a1*a3*y[n-3]/(a6*a5*a2)-  a1*a2*y[n-2]/(a4*a5*a3)+ y[n-1]*(1.0/a1+1.0/a2+1.0/a3)

    return d1


#fresnel_coef() - computes Fresnel reflectance coefficient for specified index of refr.
def fresnel_coef(mu, index):
    sq=0.0
    r2=0.0
    q1=0.0
    sq = np.sqrt(np.power(index,2.0)-1.0+np.power(mu,2.0))
    r2 = np.power((mu-sq)/(mu+sq),2.0)
    q1 = (1.0-np.power(mu,2.0)-mu*sq)/(1.0-np.power(mu,2.0)+mu*sq)

    return(r2*(q1*q1+1.0)/2.0)

#over turn a np.array
def overturn(ff):
    rows = ff.shape[0]
    dd = np.zeros_like(ff)
    for r in range(rows):
        dd[r,:] = ff[rows-1-r,:]
    return dd







