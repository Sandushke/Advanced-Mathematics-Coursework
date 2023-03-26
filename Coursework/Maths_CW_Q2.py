# Question 2
"""" Aliasing occurs when the sampled signal is not sampled at a high enough rate, which could result in overlapping
frequency components in the frequency domain(the frequency would be twice the highest frequency present in the signal).
When DFT is implemented on the sampled signal, the higher frequency in the signal gets 'aliased' to lower frequency
which would lead to incorrect results
"""
import numpy as np
import matplotlib.pyplot as plt

# Get a signal with with high frequency
sampleRate = 50
time = 1/sampleRate

range = np.arange(0, 1, time)

y = np.sin(2 * np.pi * 10 * range) + np.sin(2 * np.pi * 15 * range)

#DFT signal
x = np.fft.fft(y)

#frequency axis
freq = np.fft.fftfreq(len(y), time)

#Plot the DFT graph
plt.plot(freq, np.abs(x))
plt.xlim(0, 20)
plt.xlabel("Frequency in Hz")
plt.xlabel("Magnitude of DFT")
plt.show()

"""" In the above code, two frequency signals were generated with 10Hz and 50Hz.  When the numpy.ffy.fft is used, 
it needs to plot a graph with two peaks DFT magnitude at frequencies of 10 and 50Hz. But because of aliasing, it will
show some extra peaks in other frequencies(change in frequency values) if the sample rate is not high enough. Therefore, 
it concludes that if the sampling rate is not accurate to represent the high frequency of the signals, the result would aliase
"""

