# -*- coding: utf-8 -*-
"""
qubitServer to control all instruments

Created on 2020.09.09 20:57
@author: Tao Ziyu
"""

from functools import wraps
import logging
import numpy as np 

from zilabrad.instrument.zurichHelper import _mpAwg_init
from zilabrad.instrument import waveforms

import labrad
from labrad.units import Unit,Value
_unitSpace = ('V','mV','us','ns','s','GHz','MHz','kHz','Hz','dBm','rad','None')
V, mV, us, ns,s, GHz, MHz,kHz,Hz, dBm, rad,_l  = [Unit(s) for s in _unitSpace]

cxn = labrad.connect()
dv = cxn.data_vault


_noisy_printData = True

def loadQubits(sample, write_access=False):
    """Get local copies of the sample configuration stored in the labrad.registry.
    
    If you do not use labrad, you can create a class as a wrapped dictionary, 
    which is also saved as files in your computer. 
    The sample object can also read, write and update the files

    Returns the local sample config, and also extracts the individual
    qubit configurations, as specified by the sample['config'] list.  If
    write_access is True, also returns the qubit registry wrappers themselves,
    so that updates can be saved back into the registry.
    """
    Qubits = [sample[q] for q in sample['config']]
    sample = sample.copy()
    qubits = [sample[q] for q in sample['config']]
    
    # only return original qubit objects if requested
    if write_access:
        return sample, qubits, Qubits
    else:
        return sample, qubits






def dataset_create(dataset,dv=dv):
    """Create the dataset. 
    see 
    dataset = sweeps.prepDataset(*args)
    dv = labrad.connect().dataVault
    dataVault script is in "/server/py3_data_vault"
    """
    dv.cd(dataset.path, dataset.mkdir)
    logging.info(dataset.dependents)
    logging.info(dataset.independents)
    dv.new(dataset.name, dataset.independents, dataset.dependents)
    if len(dataset.params):
        dv.add_parameters(tuple(dataset.params))



def RunAllExperiment(exp_devices,function,iterable,
                     collect: bool = True,
                     raw: bool = False):
    """ prepare parameters: scanning axes, dependences and other parameters
    start experiment with special sequences according to the parameters
    
    exp_devices: (list/tuple): instances used to control device
    function: give special sequences, parameters, and start exp_devices, that are to be called by run()
    iterable: iterated over to produce values that are fed to function as parameters.
    collect: if True, collect the result into an array and return it; else, return an empty list
    """

    def Unit2SI(a):
        if type(a) is not Value:
            return a
        elif a.unit in ['GHz','MHz']:
            return a['Hz']
        elif a.unit in ['ns','us']:
            return a['s']
        else:
            return a[a.unit] 

    def Unit2num(a):
        if type(a) is not Value:
            return a
        else:
            return a[a.unit] 

    def run(function, paras):
        # pass in all_paras to the function
        all_paras = [Unit2SI(a) for a in paras[0]]
        swept_paras = [Unit2num(a) for a in paras[1]]
        result = function(exp_devices,all_paras)
        if raw:
            result_raws = np.asarray(result)
            for result_raw in result_raws.T:
                dv.add(result_raw)
            return result_raws
        else:
            data_send = swept_paras + result

        dv.add(data_send.copy()) ## save value to dataVault
        if _noisy_printData == True:
            print(data_send)
        return result

    
    result_list = []    
    for paras in iterable:
        result = run(function,paras)
        if collect:
            result_list.append(result)
    result_list = np.asarray(result_list)
    return result_list



def runQubits(qubits,exp_devices):
    """ generally for running multiqubits

    Args:
        qubits (list): a list of dictionary
        _runQ_servers (list/tuple): instances used to control device

    """
    qa,hd,mw,mw_r = exp_devices


    ## reload new waveform in this runQ
    for q in qubits: ## 多比特还需要修改这个send方法; 
        ## 结束hd脉冲,开始设置读取部分
        w_qa,w_hd = _mpAwg_init(q,qa,hd,mw,mw_r)
        
        
        q.xy_array = [w_hd.func2array((q.xy)[i]) for i in [0,1]]

        q.z_array = [w_hd.func2array(q.z[0])]

        def bias(self,amp=0,length=None):
            if length != None:
                self.bias_sample = ceil(length*self.fs/16)*16
            # pulse = [amp,self.bias_len]
            return np.ones(self.sample_number)*amp

        q.dc = waveforms.square(amp=q['bias'],start=w_hd.origin,end=w_hd.tlist[-1])
        q.dc_array = [w_hd.func2array(q.dc)]

        if 'do_readout' in q.keys():
            q.demod_phase = q.qa_adjusted_phase[Hz]*(qa.adc_trig_delay_s) ## adjusted phase -> unit[MHz]
            q.r = w_qa.readout([q])

        if hd.pulse_length_s != q['experiment_length']:
            hd.pulse_length_s = q['experiment_length']
            qa.set_adc_trig_delay(q['bias_start'][s]+hd.pulse_length_s)

        mw_r.set_freq(q['readout_mw_fc'])
        mw_r.set_power(q['readout_mw_power'])

        mw.set_freq(q['xy_mw_fc'])
        mw.set_power(q['xy_mw_power'])

        hd.send_waveform(waveform=q.xy_array+q.dc_array+q.z_array,
                         ports=[q.channels['xy_I'],q.channels['xy_Q'],q.channels['dc'],q.channels['z']]
                         )

        qa.send_waveform(waveform=q.r)
    ## start to run experiment
    hd.awg_open()
    qa.awg_open()
    data = qa.get_data()
    return data
