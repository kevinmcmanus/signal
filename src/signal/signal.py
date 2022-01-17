import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from src.signal.pulse import Pulse


class Signal():
    def __init__(self, tsamp = 1./1024.):

        self.tsamp = tsamp # seconds presumably
    

    def _gaussian_noise(self, sigdur, noiseprops):
        """
        sigdur in seconds
        """
        nsamp = int(sigdur/self.tsamp)
        n_sig = noiseprops['noise_sigma']
        n_mu =  noiseprops['noise_mu']

        seed = getattr(noiseprops, 'random_state', None)
        if seed is not None:
            np.random.seed(seed)

        noise = np.random.normal(n_mu, n_sig, nsamp)

        return noise

    def _sinusoid(self, sigdur, amp, freq, phase):

        nsamp = sigdur/self.tsamp
        t = np.arange(nsamp)
        sig = amp*np.sin(2*np.pi*(freq*self.tsamp)*t + phase)
        return sig


    def plot_signal(self, sig, hax=None, plot_width=4):
        """
        plot_width: int; number of seconds of the signal to plot
        """

        if hax is None:
            fig, ax = plt.subplots()
        else:
            ax = hax

        t = np.arange(len(sig))*self.tsamp
        ax.plot(t, sig)

        #for i in range(plot_width):
        #    ax.axvline(sample_rate*i, ls=':', color='red')

        ax.set_xticks([i for i in range(plot_width+1)])
        ax.set_xticklabels([str(i) for i in range(plot_width+1)])

        #quarter second minor ticks
        ax.xaxis.set_minor_locator(MultipleLocator(1/4))

        #ax.tick_params(which='both', width=2)
        ax.tick_params(which='minor', length=3)
        ax.tick_params(which='major', length=7)
        ax.xaxis.grid(True, which='major')
        ax.set_xlim(0, plot_width)
        ax.set_xlabel('Time (Seconds)')

        return ax

    def sig_gen(self, sigdur, pulse, noiseprops):
        """
        pulse_period: float, pulse period in seconds
        pulse_width: float, percentage of pulse period that the pulse occupies
        sig_dur: signal duration in seconds
        phase: where (in radians) pulse occurs in period
        peaks: whether pulses go up or down
        pulse_height: float, hieght of peaks from zero
        bias: amount by which to move the signal up or down
        noise_signma: standard deviation of noise component
        pulse_type: profile of pulse: gaussian, triangle or square
        pulse_sigma: how wide pulse should be in pulse_width (only for gaussian profile)
        """

        #generate the pulse signal
        sig = pulse.pulse_signal(tsamp=self.tsamp, sigdur=sigdur)

        # tack on the requisite noise, biased
        sig += self._gaussian_noise(sigdur, noiseprops)
        return sig