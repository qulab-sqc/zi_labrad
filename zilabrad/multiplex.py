# -*- coding: utf-8 -*-
"""
Controller for Zurich Instruments

The daily experiments are implemented via various
function (s21_scan, ramsey...)

Created on 2020.09.09 20:57
@author: Huang Wenhui, Tao Ziyu
"""


import logging  # python standard module for logging facility
import time  # show total time in experiments
import matplotlib.pyplot as plt  # give picture
from functools import wraps, reduce
import functools
import numpy as np
from numpy import pi
import itertools
import gc

from zilabrad import interface


import zilabrad.waveforms as waveforms
from zilabrad.pyle.envelopes import Envelope, NOTHING
from zilabrad.plots import dataProcess

from zilabrad.pyle import sweeps
from zilabrad.interface import gridSweep
from zilabrad.pyle.util import sweeptools
from zilabrad.instrument.qubitServer import RunAllExperiment as RunAllExp

from labrad.units import Unit, Value
import labrad
_unitSpace = ('V', 'mV', 'us', 'ns', 's', 'GHz',
              'MHz', 'kHz', 'Hz', 'dBm', 'rad', 'None')
V, mV, us, ns, s, GHz, MHz, kHz, Hz, dBm, rad, _l = [
    Unit(s) for s in _unitSpace]
ar = sweeptools.RangeCreator()


# function from interface
SQC = interface.base()
expfunc_decorator = interface.expfunc_decorator


def power2amp(power):
    """
    convert 'dBm' to 'V' for 50 Ohm (only value)
    Based on --> 0.5*Vpp=sqrt(2)*sqrt(50e-3*10^(dBm/10));
    """
    if hasattr(power, 'unit'):
        _power = power[power.unit]
    else:
        _power = power
    return 10**(_power/20-0.5)


def amp2power(amp):
    """
    convert 'dBm' to 'V' for 50 Ohm (only value)
    Based on --> 0.5*Vpp=sqrt(2)*sqrt(50e-3*10^(dBm/10));
    """
    return 20*np.log10(_amp)+10


def set_qubitsDC(qubits, experiment_length):
    for _qb in qubits:
        _qb['experiment_length'] = experiment_length
        DCbiasPulse(_qb)
    return


def XYnothing(q):
    return [NOTHING, NOTHING]


def addXYgate(
        q, start, theta, phi, piAmp='piAmp', fq='f10',
        piLen='piLen'):
    sb_freq = (q[fq] - q['xy_mw_fc'])['Hz']
    phi_t = start*sb_freq*2.*np.pi + phi
    if theta < 0:
        phi_t += np.pi
    amp = q[piAmp]*np.abs(theta)/np.pi
    length = q[piLen]['s']
    if 'xy' not in q:
        q['xy'] = XYnothing(q)
    q['xy'][0] += waveforms.cosine(amp=amp, freq=sb_freq,
                                   start=start, length=length, phase=phi_t)
    q['xy'][1] += waveforms.sine(amp=amp, freq=sb_freq,
                                 start=start, length=length, phase=phi_t)
    return


def correct_zero(sample):
    sample, qubits, Qubits = loadQubits(sample, write_access=True)
    qc = qubitContext()
    corr = qc.zero_correction
    for qubit in qubits:
        corr.correct_xy(qubit)


@expfunc_decorator
def s21_scan(sample, measure=0, stats=1024, freq=6.0*GHz, delay=0*ns, phase=0,
             mw_power=None, bias=None, power=None, zpa=0.0,
             name='s21_scan', des=''):
    """
    s21 scanning
    Args:
        sample: select experimental parameter from registry;
        stats: Number of Samples for one sweep point;
    """
    SQC.init(sample)
    qubits = SQC.qubits
    q = qubits[measure]
    q.stats = stats
    if freq is None:
        freq = q['readout_freq']
    if bias is None:
        bias = q['bias']
    if power is None:
        power = q['readout_amp']
    if mw_power is None:
        mw_power = q['readout_mw_power']

    demod_freq = q['readout_freq']-q['readout_mw_fc']
    # SQC.increase_max_length(qubits, add_len=np.max(delay))
    # add max length of hd waveforms

    axes = [(freq, 'freq'), (bias, 'bias'), (zpa, 'zpa'), (power, 'power'),
            (mw_power, 'mw_power'), (delay, 'delay'), (phase, 'phase')]
    deps = [('Amplitude', 's21 for', 'a.u.'), ('Phase', 's21 for', 'rad'),
            ('I', '', ''), ('Q', '', '')]
    kw = {'stats': stats}

    def runSweeper(devices, para_list):
        freq, bias, zpa, power, mw_power, delay, phase = para_list
        q['bias'] = bias
        q['readout_amp'] = power
        q['readout_freq'] = freq
        SQC.set_Value(qubits, 'readout_mw_fc', freq - demod_freq)

        SQC.gate_init(qubits)
        SQC.add_Z_square(q, amp=zpa, length=q.piLen)
        SQC.add_XY_gate(q, theta=0., phi=0.)
        SQC.start += q.piLen + 50*ns

        SQC.set_readout([q])
        SQC.set_DC(qubits)
        data = SQC.measure(qubits)
        return data_tools.process_IQ(data[0], q)

    SQC.prepare_dataset(
        name+des, axes, deps, kw=kw, measure=measure)
    resultArray = SQC.run_grid(runSweeper)


@expfunc_decorator
def spectroscopy(
        sample, measure=0, stats=1024, freq=None, specLen=1*us, specAmp=0.05,
        sb_freq=None, bias=None, zpa=None, name='spectroscopy', des='',
        back=False):
    def set_default(x, x_default):
        return x_default if x is None else x
    SQC.init(sample)
    qubits = SQC.qubits
    q = qubits[measure]
    SQC.set_Value(qubits, 'stats', stats)

    freq = set_default(freq, q['f10'])
    bias = set_default(bias, q['bias'])
    zpa = set_default(zpa, q['zpa'])
    specAmp = set_default(specAmp, q['piAmp'])
    specLen = set_default(specLen, q['piLen'])
    sb_freq = set_default(sb_freq, q['f10'] - q['xy_mw_fc'])

    axes = [(freq, 'freq'), (specAmp, 'specAmp'),
            (specLen, 'specLen'), (bias, 'bias'), (zpa, 'zpa')]
    deps = dependents_1q()
    kw = {'stats': stats, 'sb_freq': sb_freq}

    def runSweeper(devices, para_list):
        freq, specAmp, specLen, bias, zpa = para_list
        q['bias'] = bias
        q['piAmp'] = specAmp
        q['piLen'] = specLen
        q['f10'] = freq
        SQC.set_Value(qubits, 'xy_mw_fc', freq-sb_freq)
        start = 0

        SQC.gate_init(qubits)
        SQC.add_Z_square(q, amp=zpa, length=q.piLen+50*ns)
        SQC.add_XY_gate(q, theta=np.pi, phi=0.)
        SQC.start += q.piLen + 50*ns

        SQC.set_readout([q])
        SQC.set_DC(qubits)
        data = SQC.measure(qubits)
        return processData_1q(data[0], q)

    SQC.prepare_dataset(
        name+des, axes, deps, kw=kw, measure=measure)
    resultArray = SQC.run_grid(runSweeper)


@expfunc_decorator
def rabihigh(sample, measure=0, stats=1024, piamp=None, piLen=None, df=0*MHz,
             bias=None, zpa=None, name='rabihigh', des=''):
    """
        sample: select experimental parameter from registry;
        stats: Number of Samples for one sweep point;
    """
    def set_default(x, x_default):
        return x_default if x is None else x

    SQC.init(sample)
    qubits = SQC.qubits
    q = qubits[measure]
    SQC.set_Value(qubits, 'stats', stats)
    bias = set_default(bias, q['bias'])
    zpa = set_default(zpa, q['zpa'])
    piamp = set_default(piamp, q['piAmp'])
    piLen = set_default(piLen, q['piLen'])

    axes = [(bias, 'bias'), (zpa, 'zpa'), (df, 'df'),
            (piamp, 'piAmp'), (piLen, 'piLen')]
    deps = dependents_1q()
    kw = {'stats': stats}

    xy_mw_fc = q['xy_mw_fc']
    f10 = q['f10']

    def runSweeper(devices, para_list):
        bias, zpa, df, piamp, piLen = para_list
        q['bias'] = bias
        q['piAmp'] = piamp
        q['f10'] = f10 + df
        SQC.set_Value(qubits, 'xy_mw_fc', xy_mw_fc + df)

        SQC.gate_init(qubits)
        SQC.add_Z_square(q, amp=zpa, length=q.piLen)
        SQC.add_XY_gate(q, theta=np.pi, phi=0.)
        SQC.start += q.piLen + 50*ns

        SQC.set_readout([q])
        SQC.set_DC(qubits)
        data = SQC.measure(qubits)
        return processData_1q(data[0], q)

    SQC.prepare_dataset(
        name+des, axes, deps, kw=kw, measure=measure)
    resultArray = SQC.run_grid(runSweeper)


@expfunc_decorator
def rabihigh21(
        sample, measure=0, stats=1024, piamp21=None, piLen21=None, df=0*MHz,
        zpa=None, name='rabihigh21', des=''):
    def set_default(x, x_default):
        return x_default if x is None else x

    SQC.init(sample)
    qubits = SQC.qubits
    q = qubits[measure]
    SQC.set_Value(qubits, 'stats', stats)
    zpa = set_default(zpa, q['zpa'])
    piamp21 = set_default(piamp21, q['piAmp21'])
    piLen21 = set_default(piLen21, q['piLen21'])

    axes = [(zpa, 'zpa'), (df, 'df'),
            (piamp21, 'piamp21'), (piLen21, 'piLen21')]

    deps = dependents_1q()[:-1]
    deps_pro = [('pro', str(i), '') for i in range(3)]
    deps.extend(deps_pro)
    kw = {'stats': stats}

    f21 = q['f21']

    def runSweeper(devices, para_list):
        zpa, df, piamp21, piLen21 = para_list
        q['piAmp21'] = piamp21
        q['piLen21'] = piLen21
        q['f21'] = f21 + df

        SQC.gate_init(qubits)
        SQC.add_Z_square(q, amp=zpa, length=q.piLen+q.piLen21)
        SQC.add_XY_gate(q, theta=np.pi, phi=0.)
        SQC.start += q['piLen'] + 50*ns
        SQC.add_XY_gate(
            q, theta=np.pi, phi=0., piAmp='piAmp21', fq='f21',
            piLen='piLen21')
        start += q['piLen21']

        SQC.set_readout([q])
        SQC.set_DC(qubits)
        data = SQC.measure(qubits)

        amp, phase, Iv, Qv = data_tools.process_IQ(data[0], q)
        prob = tunneling([q], [data[0]], level=3)
        result = [amp, phase, Iv, Qv, prob[0], prob[1], prob[2]]
        return result

    SQC.prepare_dataset(
        name+des, axes, deps, kw=kw, measure=measure)
    resultArray = SQC.run_grid(runSweeper)


@expfunc_decorator
def IQraw(sample, measure=0, stats=16384, update=False, analyze=False, reps=1,
          name='IQ raw', des='', back=True):
    SQC.init(sample)
    qubits = SQC.qubits
    q = SQC.qubits[measure]
    Qb = SQC.Qubits[measure]
    SQC.set_Value(qubits, 'stats', stats)

    axes = [(reps, 'reps')]
    deps = [('Is', '|0>', ''), ('Qs', '|0>', ''),
            ('Is', '|1>', ''), ('Qs', '|1>', '')]
    kw = {'stats': stats}

    def runSweeper(devices, para_list):
        reps = para_list[0]
        # state 1
        SQC.gate_init(qubits)
        SQC.add_XY_gate(q, theta=np.pi, phi=0.)
        SQC.start += q.piLen + 50*ns
        SQC.set_readout(qubits)
        SQC.set_DC(qubits)
        data1 = SQC.measure(qubits)[measure]
        # state 0
        q['xy'] = SQC.XYnothing()
        data0 = SQC.measure(qubits)[measure]

        Is0 = np.real(data0)
        Qs0 = np.imag(data0)
        Is1 = np.real(data1)
        Qs1 = np.imag(data1)
        result = [Is0, Qs0, Is1, Qs1]
        return result

    collect, raw = True, True
    SQC.prepare_dataset(
        name+des, axes, deps, kw=kw, measure=measure)
    resultArray = SQC.run_grid(runSweeper, raw=True)
    data = np.asarray(resultArray[0])
    if update:
        dataProcess._updateIQraw2(
            data=data, Qb=Qb, dv=None, update=update, analyze=analyze)
    return data


@expfunc_decorator
def IQraw210(
        sample, measure=0, stats=1024, update=False, analyze=False,
        reps=1, name='IQ raw210', des='', back=True):
    sample, qubits, Qubits = loadQubits(sample, write_access=True)
    q = qubits[measure]
    Qb = Qubits[measure]

    for qb in qubits:
        qb.power_r = power2amp(qb['readout_amp']['dBm'])
        qb.demod_freq = qb['readout_freq'][Hz]-qb['readout_mw_fc'][Hz]

    q['stats'] = stats
    # set some parameters name;
    axes = [(reps, 'reps')]
    deps = []
    for i in range(3):
        deps += [('Is', str(i), ''), ('Qs', str(i), '')]
    kw = {'stats': stats}

    # create dataset
    dataset = sweeps.prepDataset(
        sample, name+des, axes, deps, kw=kw, measure=measure)

    def get_IQ(data):
        Is = np.real(data[0])
        Qs = np.imag(data[0])
        return [Is, Qs]

    def runSweeper(devices, para_list):
        reps = para_list[0]
        # |1>
        start = 0
        q.z = waveforms.square(
            amp=q.zpa[V], start=start, length=q.piLen[s]+q.piLen21[s])

        q['xy'] = XYnothing(q)
        addXYgate(q, start, theta=np.pi, phi=0.)

        start += q.piLen[s] + q.piLen21[s]
        start += q['qa_start_delay'][s]

        for _qb in qubits:
            _qb['experiment_length'] = start

        q['do_readout'] = True

        set_qubitsDC(qubits, q['experiment_length'])
        q.r = readoutPulse(q)

        # start to run experiment
        data1 = runQ(qubits, devices)

        # state 2
        start = 0
        q.z = waveforms.square(
            amp=q.zpa['V'], start=start, length=q.piLen['s']+q.piLen21['s'])

        q['xy'] = XYnothing(q)
        addXYgate(q, start, theta=np.pi, phi=0.)
        start += q.piLen['s']
        addXYgate(
            q, start, theta=np.pi, phi=0., piAmp='piAmp21', fq='f21',
            piLen='piLen21')

        start += q.piLen21['s']
        start += q['qa_start_delay'][s]

        data2 = runQ(qubits, devices)

        # state 0
        q.xy = [waveforms.square(amp=0), waveforms.square(amp=0)]
        data0 = runQ(qubits, devices)

        result = []
        for data in [data0, data1, data2]:
            result += get_IQ(data)
        clear_waveforms(qubits)
        return result

    collect, raw = True, True
    axes_scans = gridSweep(axes)
    results = RunAllExp(runSweeper, axes_scans, dataset, collect, raw)
    data = np.asarray(results[0])
    if update:
        adjuster.IQ_center_multilevel(qubit=Qb, data=data)
    return data


@expfunc_decorator
def measureFidelity(
        sample, rep=10, measure=0, stats=1024, update=True,
        analyze=False, name='measureFidelity', des='', back=True):
    reps = np.arange(rep)

    sample, qubits, Qubits = loadQubits(sample, write_access=True)
    q = qubits[measure]
    Qb = Qubits[measure]
    q.channels = dict(q['channels'])

    for qb in qubits:
        qb.demod_freq = qb['readout_freq'][Hz]-qb['readout_mw_fc'][Hz]

    q.sb_freq = (q['f10'] - q['xy_mw_fc'])[Hz]

    # set some parameters name;
    axes = [(reps, 'reps')]

    def deps_text(idxs): return ('measure |%d>, prepare (|%d>)' %
                                 (idxs[0], idxs[1]), '', '')
    deps = list(map(deps_text, itertools.product([0, 1], [0, 1])))

    kw = {'stats': stats}

    # create dataset
    dataset = sweeps.prepDataset(
        sample, name+des, axes, deps, kw=kw, measure=measure)

    def runSweeper(devices, para_list):
        # with pi pulse --> |1> ##
        reps = para_list
        start = 0
        q.z = waveforms.square(
            amp=q.zpa[V], start=start, length=q.piLen[s]+100e-9)
        start += 50e-9

        q.xy = XYnothing(q)
        addXYgate(q, start, np.pi, 0.)

        start += q['piLen']['s'] + 50e-9
        start += 100e-9
        start += q['qa_start_delay'][s]

        for _q in qubits:
            _q.r = readoutPulse(_q)
            _q['experiment_length'] = start
            _q['do_readout'] = True
        set_qubitsDC(qubits, q['experiment_length'])

        result = []
        # start to run experiment
        data1 = runQ(qubits, devices)[measure]

        # no pi pulse --> |0> ##
        q.xy = XYnothing(q)
        addXYgate(q, start, 0., 0.)

        # start to run experiment
        data0 = runQ(qubits, devices)[measure]

        prob0 = tunneling([q], [data0], level=2)
        prob1 = tunneling([q], [data1], level=2)
        clear_waveforms(qubits)
        return [prob0[0], prob1[0], prob0[1], prob1[1]]

    axes_scans = gridSweep(axes)
    results = RunAllExp(runSweeper, axes_scans, dataset)
    if update:
        Qb['MatRead'] = np.mean(results, 0)[1:].reshape(2, 2)
    if back:
        return results


@expfunc_decorator
def T1_visibility(sample, measure=0, stats=1024, delay=0.8*us,
                  zpa=None, bias=None,
                  name='T1_visibility', des='', back=False):
    """ sample: select experimental parameter from registry;
        stats: Number of Samples for one sweep point;
    """
    sample, qubits, Qubits = loadQubits(sample, write_access=True)
    q = qubits[measure]
    for qb in qubits:
        qb.channels = dict(qb['channels'])

    if bias is None:
        bias = q['bias']
    if zpa is None:
        zpa = q['zpa']
    q.power_r = power2amp(q['readout_amp']['dBm'])
    q.demod_freq = q['readout_freq'][Hz]-q['readout_mw_fc'][Hz]
    q.sb_freq = (q['f10'] - q['xy_mw_fc'])[Hz]

    for qb in qubits:
        qb['awgs_pulse_len'] += np.max(delay)  # add max length of hd waveforms

    # set some parameters name;
    axes = [(bias, 'bias'), (zpa, 'zpa'), (delay, 'delay')]
    deps = [('Amplitude', '1', 'a.u.'),
            ('Phase', '1', 'rad'),
            ('prob with pi pulse', '|1>', ''),
            ('Amplitude', '0', 'a.u.'),
            ('Phase', '0', 'rad'),
            ('prob without pi pulse', '|1>', '')]

    kw = {'stats': stats}

    # create dataset
    dataset = sweeps.prepDataset(
        sample, name+des, axes, deps, kw=kw, measure=measure)

    def runSweeper(devices, para_list):
        bias, zpa, delay = para_list
        # ## set device parameter

        # ----- with pi pulse ----- ###
        start = 0
        q['bias'] = bias

        q.z = waveforms.square(amp=zpa, start=start,
                               length=delay+q.piLen[s]+100e-9)
        start += 10e-9

        q.xy = XYnothing(q)
        addXYgate(q, start, np.pi, 0.)

        start += q.piLen['s'] + delay
        start += q['qa_start_delay']['s']

        q['experiment_length'] = start
        set_qubitsDC(qubits, q['experiment_length'])
        q['do_readout'] = True
        q.r = readoutPulse(q)

        # start to run experiment
        data1 = runQ(qubits, devices)
        # analyze data and return
        _d_ = data1[0]
        # unit: dB; only relative strength;
        amp1 = np.mean(np.abs(_d_))/q.power_r
        phase1 = np.mean(np.angle(_d_))
        prob1 = tunneling([q], [_d_], level=2)

        # ----- without pi pulse ----- ###
        q.xy = XYnothing(q)
        # start to run experiment
        data0 = runQ(qubits, devices)
        _d_ = data0[0]
        # analyze data and return
        amp0 = np.abs(np.mean(_d_))/q.power_r
        phase0 = np.angle(np.mean(_d_))
        prob0 = tunneling([q], [_d_], level=2)

        # multiply channel should unfold to a list for return result
        result = [amp1, phase1, prob1[1], amp0, phase0, prob0[1]]
        clear_waveforms(qubits)
        return result

    axes_scans = gridSweep(axes)
    result_list = RunAllExp(runSweeper, axes_scans, dataset)


@expfunc_decorator
def ramsey(sample, measure=0, stats=1024, delay=ar[0:10:0.4, us],
           repetition=1, df=0*MHz, fringeFreq=10*MHz, PHASE=0,
           name='ramsey', des='', back=False):
    """ sample: select experimental parameter from registry;
        stats: Number of Samples for one sweep point;
    """
    sample, qubits, Qubits = loadQubits(sample, write_access=True)
    q = qubits[measure]

    q.power_r = power2amp(q['readout_amp']['dBm'])
    q.demod_freq = q['readout_freq'][Hz]-q['readout_mw_fc'][Hz]
    q.sb_freq = (q['f10'] - q['xy_mw_fc'])[Hz]
    for qb in qubits:
        qb['awgs_pulse_len'] += np.max(delay)
    # set some parameters name;
    axes = [(repetition, 'repetition'), (delay, 'delay'), (df, 'df'),
            (fringeFreq, 'fringeFreq'), (PHASE, 'PHASE')]
    deps = [('Amplitude', 's21 for', 'a.u.'), ('Phase', 's21 for', 'rad'),
            ('I', '', ''), ('Q', '', ''), ('prob |1>', '', '')]

    kw = {'stats': stats,
          'fringeFreq': fringeFreq}

    # create dataset
    dataset = sweeps.prepDataset(
        sample, name+des, axes, deps, kw=kw, measure=measure)

    q_copy = q.copy()

    def runSweeper(devices, para_list):
        repetition, delay, df, fringeFreq, PHASE = para_list
        # set device parameter
        q['xy_mw_fc'] = q_copy['xy_mw_fc'] + df*Hz
        # ----- begin waveform ----- ###
        start = 0
        q.z = waveforms.square(
            amp=q.zpa[V], start=start, length=delay+2*q.piLen[s]+100e-9)
        start += 50e-9
        q.xy = XYnothing(q)
        addXYgate(q, start, theta=np.pi/2., phi=0.)

        start += delay + q.piLen['s']

        addXYgate(q, start, theta=np.pi/2., phi=PHASE+fringeFreq *
                  delay*2.*np.pi)

        start += q.piLen[s] + 50e-9
        start += 100e-9 + q['qa_start_delay'][s]

        q['experiment_length'] = start
        set_qubitsDC(qubits, q['experiment_length'])
        q['do_readout'] = True
        q.r = readoutPulse(q)

        # start to run experiment
        data = runQ(qubits, devices)
        # analyze data and return
        _d_ = data[0]
        # unit: dB; only relative strength;
        amp = np.abs(np.mean(_d_))/q.power_r
        phase = np.angle(np.mean(_d_))
        Iv = np.mean(np.real(_d_))
        Qv = np.mean(np.imag(_d_))
        prob = tunneling([q], [_d_], level=2)

        # multiply channel should unfold to a list for return result
        result = [amp, phase, Iv, Qv, prob[1]]
        clear_waveforms(qubits)
        return result

    axes_scans = gridSweep(axes)
    result_list = RunAllExp(runSweeper, axes_scans, dataset)


@expfunc_decorator
def s21_dispersiveShift(
        sample, measure=0, stats=1024, freq=ar[6.4:6.5:0.02, GHz],
        delay=0*ns, mw_power=None, bias=None, power=None, sb_freq=None,
        name='s21_disperShift', des='', back=False):
    """
        sample: select experimental parameter from registry;
        stats: Number of Samples for one sweep point;
    """
    sample, qubits, Qubits = loadQubits(sample, write_access=True)
    q = qubits[measure]
    q.channels = dict(q['channels'])

    if bias is None:
        bias = q['bias']
    if power is None:
        power = q['readout_amp']
    if sb_freq is None:
        sb_freq = q['readout_freq'][Hz]-q['readout_mw_fc'][Hz]
    if mw_power is None:
        mw_power = q['readout_mw_power']
    q.awgs_pulse_len += np.max(delay)  # add max length of hd waveforms
    q.demod_freq = q['readout_freq'][Hz]-q['readout_mw_fc'][Hz]

    # set some parameters name;
    axes = [(freq, 'freq'), (bias, 'bias'), (power, 'power'),
            (sb_freq, 'sb_freq'), (mw_power, 'mw_power'), (delay, 'delay')]
    deps = [('Amplitude|0>', 'S11 for %s' % q.__name__, ''),
            ('Phase|0>', 'S11 for %s' % q.__name__, rad)]
    deps.append(('I|0>', '', ''))
    deps.append(('Q|0>', '', ''))

    deps.append(('Amplitude|1>', 'S11 for %s' % q.__name__, rad))
    deps.append(('Phase|1>', 'S11 for %s' % q.__name__, rad))
    deps.append(('I|1>', '', ''))
    deps.append(('Q|1>', '', ''))
    deps.append(('SNR', '', ''))

    kw = {'stats': stats}

    # create dataset
    dataset = sweeps.prepDataset(
        sample, name+des, axes, deps, kw=kw, measure=measure)

    def runSweeper(devices, para_list):
        freq, bias, power, sb_freq, mw_power, delay = para_list
        q['readout_amp'] = power*dBm
        q.power_r = power2amp(power)

        # set microwave source device
        q['readout_mw_fc'] = (freq - q['demod_freq'])*Hz

        # write waveforms
        # with pi pulse --> |1> ##
        q.xy_sb_freq = (q['f10'] - q['xy_mw_fc'])[Hz]
        start = 0
        q.z = waveforms.square(
            amp=q.zpa[V], start=start, length=q.piLen[s]+100e-9)
        start += 50e-9
        q.xy = [waveforms.cosine(
            amp=q.piAmp, freq=q.xy_sb_freq, start=start,
            length=q.piLen[s]),
            waveforms.sine(
            amp=q.piAmp, freq=q.xy_sb_freq, start=start,
            length=q.piLen[s])]
        start += q.piLen[s] + 50e-9
        q['bias'] = bias

        start += delay
        start += 100e-9
        start += q['qa_start_delay'][s]

        q['experiment_length'] = start
        set_qubitsDC(qubits, q['experiment_length'])
        q['do_readout'] = True
        q.r = readoutPulse(q)

        # start to run experiment
        data1 = runQ([q], devices)

        # no pi pulse --> |0> ##
        q.xy = [waveforms.square(amp=0), waveforms.square(amp=0)]

        # start to run experiment
        data0 = runQ([q], devices)

        # analyze data and return
        _d_ = data0[0]
        # unit: dB; only relative strength;
        amp0 = np.abs(np.mean(_d_))/q.power_r
        phase0 = np.angle(np.mean(_d_))
        Iv0 = np.mean(np.real(_d_))
        Qv0 = np.mean(np.imag(_d_))

        _d_ = data1[0]
        # unit: dB; only relative strength;
        amp1 = np.abs(np.mean(_d_))/q.power_r
        phase1 = np.angle(np.mean(_d_))
        Iv1 = np.mean(np.real(_d_))
        Qv1 = np.mean(np.imag(_d_))
        # multiply channel should unfold to a list for return result
        result = [amp0, phase0, Iv0, Qv0]
        result += [amp1, phase1, Iv1, Qv1]
        result += [np.abs((Iv1-Iv0)+1j*(Qv1-Qv0))]
        clear_waveforms(qubits)
        return result

    axes_scans = gridSweep(axes)
    result_list = RunAllExp(runSweeper, axes_scans, dataset)
    if back:
        return result_list


##################
# Multi qubits functions
##################

def gene_binary(qNum, qLevel=2):
    import itertools
    lbs = itertools.product(np.arange(qLevel), repeat=qNum)
    labels = []
    for i, lb in enumerate(lbs):
        label = ''.join(str(e) for e in lb)
        label = '0'*(qNum-len(label)) + label
        labels += [label]
    return labels


def prep_Nqbit(qubits):
    for _q in qubits:
        # _q.power_r = power2amp(_q['readout_amp']['dBm'])
        _q.demod_freq = _q['readout_freq']['Hz']-_q['readout_mw_fc']['Hz']
        _q.sb_freq = (_q['f10'] - _q['xy_mw_fc'])['Hz']
    return


def deps_Nqbitpopu(nq: int, qLevel: int = 2):
    labels = gene_binary(nq, qLevel)
    deps = []
    for label in labels:
        deps += [('|' + label + '>', 'prob', '')]
    return deps


@expfunc_decorator
def Nqubit_state(
        sample, reps=10, measure=[0, 1], states=[0, 0],
        name='Nqubit_state', des=''):
    sample, qubits, Qubits = loadQubits(sample, write_access=True)
    reps = np.arange(reps)
    prep_Nqbit(qubits)
    axes = [(reps, 'reps')]
    labels = gene_binary(len(measure), 2)

    states_str = functools.reduce(lambda x, y: str(x)+str(y), states)
    def get_dep(x): return ('', 'measure '+str(x), '')
    deps = list(map(get_dep, labels))

    q_ref = qubits[0]
    kw = {}
    kw['states'] = states
    dataset = sweeps.prepDataset(sample, name+des, axes, deps, kw=kw)

    def runSweeper(devices, para_list):
        start = 0.
        for i, _qb in enumerate(qubits):
            _qb.xy = XYnothing(_qb)
            addXYgate(_qb, start, np.pi*states[i], 0.)

        start += max(map(lambda q: q['piLen']['s'], qubits)) + 50e-9
        for _q in qubits:
            _q.r = readoutPulse(_q)
            _q['experiment_length'] = start
            _q['do_readout'] = True

        set_qubitsDC(qubits, q_ref['experiment_length'])
        data = runQ(qubits, devices)
        prob = tunneling(qubits, data, level=2)
        clear_waveforms(qubits)
        return prob
    axes_scans = gridSweep(axes)
    result_list = RunAllExp(runSweeper, axes_scans, dataset)
    return


@expfunc_decorator
def Qstate_tomo(
        sample, rep=10, state=[0, 1], name='tomoTest',
        tbuffer=10e-9, des=''):
    sample, qubits, Qubits = loadQubits(sample, write_access=True)
    num_q = len(qubits)
    prep_Nqbit(qubits)
    reps = range(rep)
    axes = [(reps, 'reps')]
    deps = tomo_deps(num_q)
    piLens = list(map(lambda q: q['piLen'], qubits))
    kw = {'state': state}
    dataset = sweeps.prepDataset(sample, name, axes, deps, kw=kw)
    fid_matNq = read_correct_mat(qubits)

    def add_tomo_gate(qubits, idx_qst, start):
        angles = [0, np.pi/2, np.pi/2]
        phases = [0, 0, np.pi/2]
        _base3 = number_to_base(idx_qst, 3)
        idx_qst_base3 = [0]*(len(qubits)-len(_base3)) + _base3
        for i, _qt in enumerate(qubits):
            idx_pulse = idx_qst_base3[i]
            theta = angles[idx_pulse]
            phi = phases[idx_pulse]
            addXYgate(_qt, start, theta, phi)
        return

    def prepare_state(qubits, start):
        for i, q in enumerate(qubits):
            q['xy'] = XYnothing(q)
            if state[i] == 1:
                addXYgate(q, start, np.pi, 0.)
            else:
                addXYgate(q, start, 0., 0.)
        return

    def read_pulse(qubits, start):
        for _q in qubits:
            _q.r = readoutPulse(_q)
            _q['experiment_length'] = start
            _q['do_readout'] = True
        set_qubitsDC(qubits, qubits[0]['experiment_length'])
        return

    def get_prob(data):
        prob_raw = tunneling(qubits, data, level=2)
        prob = np.asarray(np.dot(fid_matNq, prob_raw))[0]
        return prob

    def runSweeper(devices, para_list):
        reps = para_list
        reqs = []
        q_ref = qubits[0]
        for idx_qst in np.arange(3**len(qubits)):
            start = 0.
            prepare_state(qubits, start)
            start += np.max(piLens)['s'] + tbuffer

            add_tomo_gate(qubits, idx_qst, start)
            start += np.max(piLens)['s'] + tbuffer
            start += q_ref['qa_start_delay']['s']

            read_pulse(qubits, start)
            data = runQ(qubits, devices)
            prob = get_prob(data)
            reqs.append(prob)
        clear_waveforms(qubits)
        return np.hstack(reqs)

    axes_scans = gridSweep(axes)
    result_list = RunAllExp(runSweeper, axes_scans, dataset)
    return


@expfunc_decorator
def qqiswap(sample, measure=0, delay=20*ns, zpa=None, name='iswap', des=''):
    """
        sample: select experimental parameter from registry;
        stats: Number of Samples for one sweep point;
    """
    sample, qubits, Qubits = loadQubits(sample, write_access=True)

    prep_Nqbit(qubits)

    q = qubits[measure]
    q_ref = qubits[0]
    if zpa is None:
        zpa = q['zpa']

    # set some parameters name;
    axes = [(delay, 'delay'), (zpa, 'zpa')]
    deps = deps_Nqbitpopu(nq=2, qLevel=2)

    # create dataset
    dataset = sweeps.prepDataset(sample, name, axes, deps, kw={})

    def runSweeper(devices, para_list):
        delay, zpa = para_list

        start = 0.

        q.xy = XYnothing(q)
        addXYgate(q, start, np.pi, 0.)

        start += q['piLen']['s'] + 50e-9

        q.z = waveforms.square(amp=zpa, start=start, length=q.piLen[s]+100e-9)

        start += 100e-9
        start += q['qa_start_delay'][s]

        for _q in qubits:
            _q.r = readoutPulse(_q)
            _q['experiment_length'] = start
            _q['do_readout'] = True

        set_qubitsDC(qubits, q_ref['experiment_length'])

        data = runQ(qubits, devices)
        prob = tunneling(qubits, data, level=2)
        clear_waveforms(qubits)
        return prob

    axes_scans = gridSweep(axes)
    result_list = RunAllExp(runSweeper, axes_scans, dataset)
    return


# ----- dataprocess tools ----- ####
class data_tools(object):
    @staticmethod
    def process_IQ(data, q):
        amp = np.abs(np.mean(data))/power2amp(q['readout_amp']['dBm'])
        phase = np.angle(np.mean(data))
        Is = np.real(np.mean(data))
        Qs = np.imag(np.mean(data))
        return [amp, phase, Is, Qs]

    @staticmethod
    def processData_1q(data, q):
        """
        process single qubit IQ data into the daily used version
        """
        amp, phase, Is, Qs = process_IQ(data, q)
        prob = tunneling([q], [data], level=2)
        return [amp, phase, Is, Qs, prob[1]]


def tunneling(qubits, data, level=2):
    """ get probability for 1,2,3...N qubits with level (2,3,4,....)
    Args:
        qubits (dict): qubit information in registry
        data (list): list of IQ data (array of complex number) for N qubits
        level (int): level of qubit
    """
    qNum = len(qubits)
    counts_num = len(data[0])
    binary_count = np.zeros((counts_num), dtype=float)

    def get_meas(data0, q, Nq=level):
        # if measure 1 then return 1
        sigs = data0

        total = len(sigs)
        distance = np.zeros((total, Nq))
        for i in np.arange(Nq):
            center_i = q['center|'+str(i)+'>'][0] + \
                1j*q['center|'+str(i)+'>'][1]
            distance_i = np.abs(sigs - center_i)
            distance[:, i] = distance_i
        tunnels = np.zeros((total,))
        for i in np.arange(total):
            distancei = distance[i]
            tunneli = np.int(np.where(distancei == np.min(distancei))[0])
            tunnels[i] = tunneli
        return tunnels

    for i in np.arange(qNum):
        binary_count += get_meas(data[i], qubits[i]) * (level**(qNum-1-i))

    res_store = np.zeros((level**qNum))
    for i in np.arange(level**qNum):
        res_store[i] = np.sum(binary_count == i)

    prob = res_store/counts_num
    return prob


def tomo_deps(num_q):
    labels = gene_binary(num_q, qLevel=2)
    deps = []
    ops = ['I', 'X', 'Y']
    for idx_qst in np.arange(len(ops)**num_q):
        op_show = ''
        for i in np.arange(num_q):
            dig_n = len(ops)
            idx_pulse = (idx_qst % (dig_n**(i+1)))//(dig_n**(i))
            op_show += ops[idx_pulse]
        for l1 in labels:
            deps.append((op_show, '_P|'+l1+'>', ''))
    return deps


def read_correct_mat(qubits):
    Q_cals = []
    for i, q in enumerate(qubits):
        Q_cals.append(np.mat(q['MatRead']))
    fid_matNq = np.mat(reduce(lambda x, y: np.kron(x, y), Q_cals)).I
    return fid_matNq


def dependents_1q():
    deps = [('Amp', 's21 for', 'a.u.'), ('Phase', 's21 for', 'rad'),
            ('I', '', ''), ('Q', '', ''),
            ('pro', '|1>', '')]
    return deps


def number_to_base(n, b):
    """
    example:
    n = 10, b = 3
    return [1, 0, 1]
    """
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]


def processData_1q(data, q):
    """
    process single qubit IQ data into the daily used version
    """
    # only relative strength;
    amp = np.abs(np.mean(data))/power2amp(q['readout_amp']['dBm'])
    phase = np.angle(np.mean(data))
    Iv = np.real(np.mean(data))
    Qv = np.imag(np.mean(data))
    prob = tunneling([q], [data], level=2)
    # multiply channel should unfold to a list for return result
    result = [amp, phase, Iv, Qv, prob[1]]
    return result

# ----------- dataprocess tools END -----------#


# ------------- old code basket --------------#

# ------------- code basket end --------------#
