from scipy.io import wavfile
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import signal
from scipy.signal import spectrogram

INPUT_NOISE_DIR = '../../data/raw_noise/'
INPUT_CLEAN_DIR = '../../data/sliced_clean/'
OUTPUT_DIR = '../../data/processed/'

if __name__ == '__main__':
	processed_data = []

	for clean in os.listdir(INPUT_CLEAN_DIR):
		if clean[-4:] == '.wav':
			rate_clean, data_clean = wavfile.read(INPUT_CLEAN_DIR + clean)
			for noise in os.listdir(INPUT_NOISE_DIR):
				if noise[-4:] == '.wav':
					
					_, data_noise = wavfile.read(INPUT_NOISE_DIR + noise)

					length = len(data_clean)	

					data_noise = data_noise[:length][:]

					data_combined = [(s1/2 + s2/2) for (s1, s2) in zip(data_clean, data_noise)]

					_, _, Sx_clean = spectrogram(data_clean, fs=rate_clean, window=('hamming'), nperseg=441)
					_, _, Sx_noise = spectrogram(data_noise, fs=rate_clean, window=('hamming'), nperseg=441)
					_, _, Sx_combined = spectrogram(data_combined, fs=rate_clean, window=('hamming'), nperseg=441)

					Sx_target = np.concatenate((Sx_clean, Sx_noise), axis=0)

					processed_data.append([Sx_combined, Sx_target])	
			
			print('Finished processing clean noise %s' % (clean))

	np.save(OUTPUT_DIR + 'train.npy', processed_data)