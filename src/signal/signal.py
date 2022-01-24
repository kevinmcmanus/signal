import numpy as np

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

    #bring in the plot routines
    from src.signal.plot import plot_spectrum, plot_signal