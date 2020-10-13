# -*- coding: utf-8 -*-
"""tested codes
"""

from functools import wraps
import logging
import time


from zilabrad.instrument import *
from zilabrad.pyle import *


_unitSpace = ('V','mV','us','ns','s','GHz','MHz','kHz','Hz','dBm','rad','None')
V, mV, us, ns,s, GHz, MHz,kHz,Hz, dBm, rad,_l = [Unit(s) for s in _unitSpace]
   
ar = sweeptools.r

def runDummy(qubits,exp_devices):
    data = np.array([])
    for q in qubits:
        if len(data) == 0:
            data_new = np.random.random(q['stats']) + 1j*np.random.random(q['stats'])
            data = data_new
        else:
            data_new = np.random.random(q['stats']) + 1j*np.random.random(q['stats'])
            data = np.vstack([data,data_new])
    return data

def RunAllDummy(*args):
    exp_devices = [None]*4
    qubitServer._noisy_printData = False
    data = qubitServer.RunAllExperiment(exp_devices,*args)
    return data

def stop_device():
    logger.info('Stop running '%(time.time()-_t0_))

def expfunc_decorator(func):
    """
    do some stuff before call the function (func) in our experiment
    do stuffs.... func(*args) ... do stuffs
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__name__)
#         check_device()
        _t0_ = time.time()
        start_ts = time.time()
        result = func(*args, **kwargs)
#         stop_device() ## stop all device running
        logger.info('use time (s): %.2f '%(time.time()-_t0_))
        return result
    return wrapper

@expfunc_decorator
def testing_runQ(sample,measure=0,stats=1024,freq=6.0*GHz,delay=0*ns,power=-30.0*dBm,sb_freq=None,
    name='virtual testing',des='',back=False):
    """ testing runQubits looping
    """
    sample, qubits, Qubits = loadQubits(sample, write_access=True)
    q = qubits[measure]
    ## set some parameters name;
    axes = [(freq,'freq'),(power,'power'),(delay,'delay')]
    deps = [('Amp','','a.u.'),('Phase','','rad'),('I','',''),('Q','','')]
    kw = {'stats': stats}

    # create dataset
    dataset = sweeps.prepDataset(sample, name+des, axes, deps,kw=kw)
    dataset_create(dataset)

    def runSweeper(devices,para_list):
        freq,power,delay = para_list
        print(power)
        q.power_r = 10**(power/20-0.5)
        data = runDummy([q],devices)
        for _d_ in data:
            amp = np.mean(np.abs(_d_))/q.power_r ## unit: dB; only relative strength;
            phase = np.mean(np.angle(_d_))
            Iv = np.mean(np.real(_d_))
            Qv = np.mean(np.imag(_d_))
        result = [amp,phase,Iv,Qv]
        return result 

    axes_scans = checkAbort(gridSweep(axes), prefix=[1],func=stop_device)
    result_list = RunAllDummy(runSweeper,axes_scans)
    if back:
        return result_list

def run(ss=None):
    if ss == None:
        import labrad
        cxn=labrad.connect()
        ss = switchSession(cxn,user='hwh',session=None)
    testing_runQ(ss,freq=ar[6.0:6.9:0.01,GHz])
    return