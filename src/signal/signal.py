import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


class Signal():
    def __init__(self, noise_sigma=1, noise_mu=0, tsamp = 1./1024.):
        self.noise_mu = noise_mu
        self.noise_sigma = noise_sigma
        self.tsamp = tsamp # seconds presumably
    
    def _pulse_profile(self, pulse_length, pulse_type='gaussian', pulse_sigma=2):
        """
        pulse_len is the number of samples that the pulse is to occupy.
        Pulse is centered in this interval
        """
        if pulse_type == 'square':
            pulse = np.ones(pulse_length)
        elif pulse_type == 'left triangle':
            pulse = np.linspace(1,0, pulse_length)
        elif pulse_type == 'right triangle':
            pulse = np.linspace(0,1, pulse_length)
        elif pulse_type == 'triangle':
            nleft = pulse_length//2
            nright = pulse_length-nleft
            pulse = np.concatenate([np.linspace(0,1, nleft, endpoint=False), np.linspace(1,0, nright)])
        elif pulse_type == 'gaussian':
            x = np.linspace(-10, 10, pulse_length)
            pulse = np.exp(-x**2/(2*(pulse_sigma**2)))
        else:
            raise ValueError(f'Invalid pulse_type: {pulse_type}')

        return pulse

    def _gaussian_noise(self, sigdur, noise_sigma=None, noise_mu=None, random_state=None):
        """
        sigdur in seconds
        """
        nsamp = int(sigdur/self.tsamp)
        n_sig = noise_sigma if noise_sigma is  not None else self.noise_sigma
        n_mu =  noise_mu if noise_mu is not None else self.noise_mu

        if random_state is not None:
            np.random.seed(random_state)

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

    def sig_gen(self, pulse_period, pulse_width = 0.05, sigdur = 300,
            phase=np.pi,  peaks = True, pulse_height=1.0,
            bias = 0.0, noise_sigma=0.05,
            pulse_type = 'gaussian', pulse_sigma=3, random_state=None):
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

        if phase < 0.0 and phase > 2*np.pi:
                raise ValueError('Phase must be between 0 and 2 pi')

        #create a single pulse period
        samples_per_pulseperiod = int(pulse_period/self.tsamp)
        samples_per_pulse = int(samples_per_pulseperiod*pulse_width)
        padleft = (samples_per_pulseperiod-samples_per_pulse)//2
        padright = samples_per_pulseperiod - samples_per_pulse - padleft
        polarity = 1 if peaks else -1
        pulse = pulse_height*polarity*self._pulse_profile(samples_per_pulse, pulse_type=pulse_type, pulse_sigma=pulse_sigma)
        sig_period = np.pad(pulse, (padleft, padright)) # pulse is in the middle of the period (at pi radians)

        #adjust the phase
        pulse_radians_per_sample = 2*np.pi/samples_per_pulseperiod
        phi = phase - np.pi # radians to offset from middle of pulse period
        phase_offset = int(phi/pulse_radians_per_sample)
        sig_period = np.roll(sig_period, phase_offset)

        # string the periods together for the duration of the signal
        nperiods = int(sigdur/pulse_period)
        sig = np.tile(sig_period, nperiods)

        # tack on the requisite noise, biased
        sig += self._gaussian_noise(len(sig)*self.tsamp, noise_sigma=noise_sigma, noise_mu=bias, random_state=random_state )

        return sig