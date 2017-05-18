from scipy.io import wavfile
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import signal
from create_wavefile import *
import hickle as hkl

INPUT_NOISE_DIR = '../../data/raw_noise/'
INPUT_CLEAN_DIR = '../../data/sliced_clean/'
OUTPUT_DIR = '../../data/processed/'

if __name__ == '__main__':
	processed_data = []

	noise_data = [wavfile.read(INPUT_NOISE_DIR + noise)[1] for noise in os.listdir(INPUT_NOISE_DIR) if noise[-4:] == '.wav']
	noise_data = noise_data[:5]

	for i, clean in enumerate(os.listdir(INPUT_CLEAN_DIR)):
		if clean[-4:] == '.wav' and (clean[:2] == 'f2'):
			rate_clean, data_clean = wavfile.read(INPUT_CLEAN_DIR + clean)
			for noise in noise_data:
				data_noise = noise[:]

				length = len(data_clean)
				data_noise = data_noise[:length][:]

				data_combined = np.array([(s1/2 + s2/2) for (s1, s2) in zip(data_clean, data_noise)])

				Sx_clean = pretty_spectrogram(data_clean.astype('float64'), fft_size=fft_size, step_size=step_size, thresh=spec_thresh)
				Sx_noise = pretty_spectrogram(data_noise.astype('float64'), fft_size=fft_size, step_size=step_size, thresh=spec_thresh)
				Sx_combined = pretty_spectrogram(data_combined.astype('float64'), fft_size=fft_size, step_size=step_size, thresh=spec_thresh)

				# Sx_target = np.concatenate((Sx_clean, Sx_noise), axis=0)

				processed_data.append([Sx_combined, Sx_clean, Sx_noise])
			print('Finished processing %d clean slice files' % (i + 1))

	# hkl.dump(processed_data, OUTPUT_DIR + 'data.hkl')
	np.save(OUTPUT_DIR + 'data.npy', processed_data)