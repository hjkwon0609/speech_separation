from scipy.io import wavfile
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import signal
from scipy.signal import spectrogram

np.set_printoptions(threshold=np.nan)


DATA_DIR = '../../data/sliced_clean/'
# OUTPUT_DIR = '../../data/sliced_clean/'


for f in os.listdir(DATA_DIR):
	# if f[-4:] == '.wav':
	if f == 'f1_script1_clean_34.wav':
		rate, data = wavfile.read(DATA_DIR + f)
		# data = signal.decimate(data, 4)
		# data *= np.hamming(len(data))

		# data = abs(np.fft.rfft(data))
		# X_db = 20 * np.log10(data)
		# freqs = np.fft.rfftfreq(441, 1.0/44100)
		# plt.plot(freqs, X_db)
		# plt.show
		f, t, Sx = spectrogram(data, fs=rate, window=('hamming'), nperseg=441)
		print Sx
		break