import numpy as np
import math

num_freq_bins = 512

frequencies = np.array([2.0 * 180 * i / num_freq_bins * 22050 / 360 for i in xrange(num_freq_bins)])
frequencies[0] = 2.0 * 180 / num_freq_bins / 2 * 22050 / 360 # 0th frequency threshold is computed at 3/4th of the frequency range
weights = 3.64 * np.power(1000 / frequencies, 0.8) - 6.5 * np.exp(-0.6 * np.power(frequencies / 1000 - 3.3, 2)) + np.power(0.1, 3) * np.power(frequencies / 1000, 4)
print(frequencies)
print(weights)
