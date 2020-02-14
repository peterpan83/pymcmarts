from utils import AhmadAerosolReader as AReader
from inputs import input
import numpy as np


sensor = 'msia'
bands = AReader.get_bands(sensor)
bands_int = np.asarray(bands,dtype=int)

print (bands)

rhs, fmfs =AReader.get_names(sensor)


for i,wl in enumerate(bands_int):
    scatt, phases = AReader.get_phase_all(sensor, 0)
    input.gen_aerosol_phase(sensor=sensor,band=wl, scatt=scatt,phases=phases,rhs=rhs,fmfs=fmfs)






