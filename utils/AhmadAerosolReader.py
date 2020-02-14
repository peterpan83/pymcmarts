
import aerosolmodle.utilities as utl
import aerosolmodle.sensorInfo as sinf
import aerosolmodle.aeromodelTab as at
import numpy as np

AEROSOL_LUT = None

def get_bands(sensorname):
    global AEROSOL_LUT
    if not AEROSOL_LUT:
        loadLUT(sensorname)
    return AEROSOL_LUT.si.wave


def get_names(sensorname):
    global AEROSOL_LUT
    if not AEROSOL_LUT:
        loadLUT(sensorname)
    return AEROSOL_LUT.rhs, AEROSOL_LUT.fmfs

def loadLUT(sensorname='oli'):
    global AEROSOL_LUT
    si = sinf.SensorInfo(sensorName=sensorname)
    si.readSensorInfo()
    AEROSOL_LUT = at.AeromodelTab(sensorName=sensorname,sensorinfo=si,ocsswdata =utl.OCSSWDATA ,modelnames=utl.AEROSOLNAMES)
    AEROSOL_LUT.load()

def get_phase(sensorname,modelindex,bandindex):
    '''
    get scatter angle and relative phase
    :param sensorname: oli
    :param modelindex: 0-79
    :param bandindex:
    :return:
    '''
    global AEROSOL_LUT
    if not AEROSOL_LUT:
        loadLUT(sensorname)
    aero_model = AEROSOL_LUT.models[modelindex]
    p = aero_model.phase[bandindex,:]
    phase_relative = p/p.sum()
    return aero_model.scatt, phase_relative


def get_phase_all(sensorname, bandindex):
    global AEROSOL_LUT
    phase = []
    if not AEROSOL_LUT:
        loadLUT(sensorname)
    for i in range(80):
        aero_model = AEROSOL_LUT.models[i]
        p = aero_model.phase[bandindex,:]
        phase_relative = p / p.sum()
        phase.append(phase_relative)
    return AEROSOL_LUT.scatt, np.asarray(phase)


def get_angstroms(sensorname, modelindex, referindex,bands):
    '''
    :param sensorname:
    :param modelindex:
    :param referindex:
    :param bands:
    :return:
    '''
    global AEROSOL_LUT
    if not AEROSOL_LUT:
        loadLUT(sensorname)
    aero_model = AEROSOL_LUT.models[modelindex]
    angstroms = []
    for i, w in  enumerate(bands):
        angstrom = aero_model.extc[i] / aero_model.extc[referindex]
        angstroms.append(angstrom)
    return np.asarray(angstroms)