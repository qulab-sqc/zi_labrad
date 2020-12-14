from functools import wraps
import time
import logging
import gc
import numpy as np

from zilabrad.instrument import QubitContext
from zilabrad.instrument import qubitServer
from zilabrad import waveforms
from zilabrad.pyle import sweeps
from zilabrad.pyle import envelopes

logger = logging.getLogger(__name__)
logger.setLevel('WARNING')


def gridSweep(axes):
    """
    gridSweep generator yield all_paras, swept_paras
    if axes has one iterator, we can do a one-dimensional scanning
    if axes has two iterator, we can do a square grid scanning

    you can also create other generator, that is conditional for the
    result, do something like machnine-learning optimizer

    Example:
    axes = [(para1,'name1'),(para2,'name2'),(para3,'name3')...]
    para can be iterable or not-iterable

    for paras in gridSweep(axes):
        all_paras, swept_paras = paras

    all_paras: all parameters
    swept_paras: iterable parameters
    """
    if not len(axes):
        yield (), ()
    else:
        (param, _label), rest = axes[0], axes[1:]
        # TODO: different way to detect if something should be swept
        if np.iterable(param):
            for val in param:
                for all, swept in gridSweep(rest):
                    yield (val,) + all, (val,) + swept
        else:
            for all, swept in gridSweep(rest):
                yield (param,) + all, swept


def expfunc_decorator(func):
    """
    do some stuff before call the function (func) in our experiment
    do stuffs.... func(*args) ... do stuffs
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_ts = time.time()
        try:
            result = func(*args, **kwargs)
        except KeyboardInterrupt:
            # stop in the middle
            logger.warning('KeyboardInterrupt')
            logger.warning('stop_device')
            timeNow = time.strftime("%Y-%m-%d %X", time.localtime())
            logger.warning(timeNow)
            qubitServer.stop_device()
        else:
            # finish in the end
            qubitServer.stop_device()
            timeNow = time.strftime("%Y-%m-%d %X", time.localtime())
            logger.info(timeNow)
            return result
    return wrapper


class base(object):
    """
    base interface for experiment workflow
    """
    def __init__(self):
        """
        Args:
            start: to mark when the gate start
        """
        self.start = 0
        self.qubits = None
        self.Qubits = None
        self.is_prepare_dataset = False

    def clear_temp(self):
        self.start = 0
        self.qubits = None
        self.Qubits = None
        gc.collect()

    def init(self, sample):
        self.clear_temp()
        sample, qubits, Qubits = QubitContext.loadQubits(
            sample, write_access=True)
        self.qubits = qubits
        self.Qubits = Qubits
        self.sample = sample

    def prepare_dataset(self, name, axes, deps, kw, measure):
        self.axes = axes
        self.deps = deps
        self.dataset = sweeps.prepDataset(
                self.sample, name,
                self.axes, self.deps,
                kw=kw, measure=measure)

        self.is_prepare_dataset = True

    def run_grid(self, function, raw=False):
        """scan paras in a grid form
        """
        if self.is_prepare_dataset is False:
            raise Exception("You have not call prepare_dataset first")
        else:
            # reset for next experiment
            self.is_prepare_dataset = False
        axes_scans = gridSweep(self.axes)
        resultArray = qubitServer.RunAllExperiment(
            function, axes_scans, self.dataset, raw=raw)
        return resultArray

    def gate_init(self, qubits):
        self.start = 0
        clear_keys = ['z', 'xy', 'dc', 'r']
        for q in qubits:
            for key in clear_keys:
                if q.get(key) is not None:
                    q.pop(key)

    @staticmethod
    def set_Value(qubits, key, value):
        for q in qubits:
            q[key] = value

    def set_DC(self, qubits: list):
        for q in qubits:
            q['experiment_length'] = self.start
            DCbiasPulse(q)

    def set_readout(self, qubits: list, start=None):
        if start is None:
            start = self.start
        for q in qubits:
            q['demod_freq'] = q['readout_freq']-q['readout_mw_fc']
        q_ref = qubits[0]
        start += q_ref['qa_start_delay']
        for q in qubits:
            q.r = readoutPulse(q)
            q['experiment_length'] = start
            q['do_readout'] = True
        return

    def increase_max_length(self, qubits, add_len):
        for q in qubits:
            q['awgs_pulse_len'] += np.max(add_len)
            # add max length of hd waveforms

    def measure(self, qubits):
        data = qubitServer.runQubits(qubits)
        # reset for safety
        self.start = 0
        return data

    @staticmethod
    def XYnothing():
        return [envelopes.NOTHING, envelopes.NOTHING]

    def add_XY_gate(
            self, q, theta, phi, piAmp='piAmp', fq='f10',
            piLen='piLen', start=None):
        if start is None:
            start = self.start
        sb_freq = (q[fq] - q['xy_mw_fc'])['Hz']
        phi_t = self.start*sb_freq*2.*np.pi + phi
        if theta < 0:
            phi_t += np.pi
        amp = q[piAmp]*np.abs(theta)/np.pi
        length = q[piLen]['s']
        if 'xy' not in q:
            q['xy'] = [envelopes.NOTHING, envelopes.NOTHING]
        q['xy'][0] += waveforms.cosine(
            amp=amp, freq=sb_freq, start=start,
            length=length, phase=phi_t)
        q['xy'][1] += waveforms.sine(
            amp=amp, freq=sb_freq, start=start,
            length=length, phase=phi_t)
        return

    def add_Z_square(self, q, amp, length, start=None):
        if start is None:
            start = self.start
        if 'z' not in q:
            q['z'] = envelopes.NOTHING
        q.z += waveforms.square(
            amp=amp, start=start, length=length)


def DCbiasPulse(q):
    if type(q['bias']) in [float, int]:
        bias = q['bias']
    else:
        bias = q['bias']['V']

    if abs(bias) > 2.5:
        raise ValueError("bias out of range (-2.5 V, 2.5 V)")
    if 'z' not in q:
        q['z'] = envelopes.NOTHING

    channels = dict(q.channels)
    start = -q['bias_start']
    end = (q['bias_end'] + q['readout_len'] +
           q['experiment_length'])

    pulse = waveforms.square(
            amp=bias, start=start, end=end
            )
    if channels.get('dc') is None:
        q['z'] += pulse
    else:
        q['dc'] = pulse


def readoutPulse(q):
    power = q['readout_amp']
    if hasattr(power, 'unit'):
        _power = power[power.unit]
    else:
        _power = power
    amp = 10**(_power/20-0.5)

    length = q['readout_len']
    # when measuring T1_visibility or others, we need start delay of readout
    return waveforms.readout(
        amp=amp, phase=q['demod_phase'],
        freq=q['demod_freq'], start=0, length=length
        )
