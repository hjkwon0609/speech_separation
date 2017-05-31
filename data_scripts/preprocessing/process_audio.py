from scipy.io import wavfile
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import signal
from create_wavefile import *
# import hickle as hkl
import stft
import h5py

INPUT_NOISE_DIR = '../../data/raw_noise/'
INPUT_CLEAN_DIR = '../../data/sliced_clean/'
OUTPUT_DIR = '../../data/processed/'

def pad_data(data):
	num_samples = len(data)
	# print(num_samples)
	max_rows_in_sample = max(len(data[i]) for i in xrange(num_samples))
	# print([len(data[i]) for i in xrange(num_samples)])
	# print(max_rows_in_sample)
	num_cols_in_row = data[0][0].size
	# print(num_cols_in_row)
	# print(data[0][0])
	new_data = np.zeros((num_samples, max_rows_in_sample, num_cols_in_row))
	for i, sample in enumerate(data):
		for j, row in enumerate(sample):
			num_rows = len(sample)
			# print(row)
			for k, c in enumerate(row):
				new_data[i][max_rows_in_sample - num_rows + j][k] = c
	return new_data

if __name__ == '__main__':
	processed_data = []

	noise_data = [wavfile.read(INPUT_NOISE_DIR + noise)[1] for noise in os.listdir(INPUT_NOISE_DIR) if noise[-4:] == '.wav']
	noise_data = noise_data[:5]

	batch_size = 200
	curr = 0
	curr_batch = 0

	for i, clean in enumerate(os.listdir(INPUT_CLEAN_DIR)):
			
		if clean[-4:] == '.wav':
			rate_clean, data_clean = wavfile.read(INPUT_CLEAN_DIR + clean)
			for noise in noise_data:
				data_noise = noise[:]

				length = len(data_clean)
				data_noise = data_noise[:length][:]

				data_combined = np.array([(s1/2 + s2/2) for (s1, s2) in zip(data_clean, data_noise)])

				Sx_clean = stft.spectrogram(data_clean).transpose() / 100000
				Sx_noise = stft.spectrogram(data_noise).transpose() / 100000
				Sx_combined = stft.spectrogram(data_combined).transpose() / 100000
				# Sx_clean = pretty_spectrogram(data_clean.astype('float64'), fft_size=fft_size, step_size=step_size, thresh=spec_thresh)
				# Sx_noise = pretty_spectrogram(data_noise.astype('float64'), fft_size=fft_size, step_size=step_size, thresh=spec_thresh)
				# Sx_combined = pretty_spectrogram(data_combined.astype('float64'), fft_size=fft_size, step_size=step_size, thresh=spec_thresh)

				# Sx_target = np.concatenate((Sx_clean, Sx_noise), axis=0)
				# print(clean)
				# print (Sx_clean.shape)

				processed_data.append([Sx_combined, Sx_clean, Sx_noise])
			
			curr_batch += 1
			if curr_batch == batch_size:
				combined, clean, noise = zip(*processed_data)
				
				combined_padded = pad_data(combined)
				clean_padded = pad_data(clean)
				noise_padded = pad_data(noise)

				processed_data = np.array([combined_padded, clean_padded, noise_padded])

				# np.savez_compressed('%sdata%d' % (OUTPUT_DIR, curr), processed_data)
				f = h5py.File('%sdata%d' % (OUTPUT_DIR, curr), 'w')
				f.create_dataset('data', data=processed_data, compression="gzip", compression_opts=9)
				print('Saved batch curr %d' % (curr))
				processed_data = []
				curr += 1
				curr_batch = 0
			
			print('Finished processing %d clean slice files' % (i + 1))

	# np.savez_compressed('%sdata' % (OUTPUT_DIR), processed_data)

	# hkl.dump(processed_data, OUTPUT_DIR + 'data.hkl')
	