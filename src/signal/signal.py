import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from scipy.stats import kstest, chi2

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


    def get_normdft(self, retval=True, recalc=False):
        sig = self.get_signal()
      
        noise_sigma = 1.0
        if not self.noiseprops is None:
            noise_sigma = self.noiseprops.get('noise_sigma', 1.0)
        

        if recalc or self.dft is None:
            self.dft = np.fft.rfft(sig, norm='ortho')*np.sqrt(2)/noise_sigma
        
        return self.dft if retval else None

    def plot_spectrum(self, ax=None, maxfreq=100, ymin=0., thresh=0.005):

            if ax is None:
                fig, ax = plt.subplots()

            dft = self.get_normdft()
            freqs = np.fft.rfftfreq(len(self.signal), d=self.tsamp)
            pwr = np.power(np.abs(dft),2)

            #calculate the significance threshold
            chi_thresh = chi2.ppf(1-thresh, 2 ) #two deg freedom for each power value
            #note on ymin:
            #The power coefs which are to be plotted are chisq(nu, 2nu) where nu=2
            #(real and imaginary parts); the variance is sigma^2=2nu=4, so sigma=2.
            #ymin =0 is therefore one stddev below the mean.


            #ditch the dc bias coefficient
            pwr = pwr[1:]
            freqs = freqs[1:]
            nsig = (pwr>chi_thresh).sum() # number of significant power values

            #find the frequency bin of max power and calculate the period from that
            max_pwr_i = pwr.argmax()
            max_pwr_freq = freqs[max_pwr_i]
            per = 1/max_pwr_freq

            ax.semilogy(freqs, pwr, ls='None', marker='o', markersize=1)
            ax.axhline(chi_thresh, color='red', lw=4, ls=':', label='Threshold')

            ax.text(0.2, 0.95, f'Max Power: {pwr[max_pwr_i]:.3e}',transform=ax.transAxes)
            ax.text(0.2, 0.90, f'Max Power Freq: {max_pwr_freq:.3f} s^-1',transform=ax.transAxes)
            ax.text(0.2, 0.85, f'Calculated Period: {per:.3f} s',transform=ax.transAxes)
            ax.text(0.2, 0.80, f'N Signif Coefs: {nsig:,}',transform=ax.transAxes)
            ax.set_xlabel('Frequency ($s^{-1}$)')
            ax.set_ylabel('Log10 Power')
            ax.set_title('Frequency Domain')
            ax.set_xlim(0,maxfreq)
            ax.set_ylim(ymin)
            ax.legend(loc='upper right')

    def plot_signal(self, ax=None, plot_width=4):
        """
        plot_width: int; number of seconds from the beginnig of the signal to plot
        """

        if ax is None:
            fig, ax = plt.subplots()

        sig = self.get_signal()

        if not self.noiseprops is None:
            noise_mu = self.noiseprops.get('noise_mu', 0)
            noise_sigma = self.noiseprops.get('noise_sigma', 1.0)
        else:
            # calculate from signal
            noise_mu = sig.mean()
            noise_sigma = sig.std(ddof=1)

        #Kolomogorov-Smirnov Test for normality:
        ks_v, ks_p = kstest(sig,'norm',[noise_mu,noise_sigma])

        t = np.arange(len(sig))*self.tsamp
        ax.plot(t, sig, lw=1)

        #for i in range(plot_width):
        #    ax.axvline(sample_rate*i, ls=':', color='red')

        ax.text(0.05, 0.95, f'KS val: {ks_v:.3f}',transform=ax.transAxes)
        ax.text(0.05, 0.90, f'KS P-val: {ks_p:.3f}',transform=ax.transAxes)

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
        ax.set_title('Time Domain')

        return ax
