import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from src.signal.pulse import Pulse


class Signal():
    def __init__(self, tsamp = 1./1024., #sampling interval
                pulseprops=None,
                sinprops=None,
                noiseprops=None ):

        self.tsamp = tsamp # seconds presumably
        self.pulseprops = pulseprops
        self.sinprops = sinprops # could be a list of sinprops
        self.noiseprops = noiseprops
        self.signal = None
        self.dft = None
    

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

    def _sinusoid(self, sigdur, sinprops ):
        if not type(sinprops) is dict and not type(sinprops) is list:
            raise ValueError('Invalid sinprops')

        if type(sinprops) is dict:
            sinprops = [sinprops]

        nsamp = int(sigdur/self.tsamp)

        t = np.arange(nsamp)
        sig = np.zeros(nsamp)
        for sin in sinprops:
            amp = sin['amplitude']
            freq = sin['frequency']
            phase = sin['phase']

            sig += amp*np.sin(2*np.pi*(freq*self.tsamp)*t + phase)

        return sig

    def _sig_gen(self, sigdur):

        self.sigdur = sigdur
        nsamp = int(sigdur/self.tsamp)
        sig = np.zeros(nsamp)

        if self.pulseprops is not None:
            pulse = Pulse(**self.pulseprops)
            #generate the pulse signal (signal )
            sig += pulse.pulse_signal(tsamp=self.tsamp, sigdur=self.sigdur)

        if self.sinprops is not None:
            sig += self._sinusoid(sigdur, self.sinprops)

        if self.noiseprops is not None:
            # tack on the requisite noise, biased
            sig += self._gaussian_noise(sigdur, self.noiseprops)
        
        return sig

    def get_signal(self, sigdur=300, retval=True, recalc=False):
        if recalc or self.signal is None:
            self.signal = self._sig_gen(sigdur)
            self.dft = None
        if retval:
            return self.signal
        else:
            return None

    def plot_signal(self, ax=None, plot_width=4):
        """
        plot_width: int; number of seconds from the beginnig of the signal to plot
        """

        if ax is None:
            fig, ax = plt.subplots()

        sig = self.get_signal()

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

    def get_normdft(self, retval=True, recalc=False):
        sig = self.get_signal()
      
        noise_sigma = 1.0
        if not self.noiseprops is None:
            noise_sigma = getattr(self.noiseprops, 'noise_sigma', 1.0)
        

        if recalc or self.dft is None:
            self.dft = np.fft.rfft(sig, norm='ortho')*np.sqrt(2)/noise_sigma
        
        return self.dft if retval else None

    def plot_spectrum(self, ax=None, xlim=None):

        if ax is None:
            fig, ax = plt.subplots()

        dft = self.get_normdft()
        freqs = np.fft.rfftfreq(len(self.signal), d=self.tsamp)
        pwr = np.power(np.abs(dft),2)

        #ditch the dc bias coefficient
        pwr = pwr[1:]
        freqs = freqs[1:]

        print(f'Siglen: {len(self.signal)}, dftlen: {len(pwr)}, freqlen {len(freqs)}')
        max_pwr_i = pwr.argmax()
        max_pwr_freq = freqs[max_pwr_i]
        per = 1/max_pwr_freq

        ax.plot(freqs, pwr)
        #ax.text(0.6, 0.8, f'Sample Rate: {sample_rate} s^-1',transform=ax.transAxes)
        #ax.text(0.6, 0.75, f'Pulse Period: {period:.3f} s',transform=ax.transAxes)
        #ax.text(0.6, 0.70, f'Pulse Width: {100*width} %',transform=ax.transAxes)
        ax.text(0.5, 0.65, f'Max Power: {pwr[max_pwr_i]:.3e}',transform=ax.transAxes)
        ax.text(0.5, 0.60, f'Max Power Freq: {max_pwr_freq:.3f} s^-1',transform=ax.transAxes)
        ax.text(0.5, 0.55, f'Calculated Period: {per:.3f} s',transform=ax.transAxes)
        ax.set_xlabel('Frequency ($s^{-1}$)')
        ax.set_xlim(xlim)
