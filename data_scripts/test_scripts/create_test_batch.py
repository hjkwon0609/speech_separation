import h5py
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os
from scipy import signal
import stft
import pickle

# def create_batch(input_data, target_data, batch_size):
# 	input_batches = []
# 	target_batches = []

# 	for i in xrange(0, len(target_data), batch_size):
# 		input_batches.append(input_data[i:i + batch_size])
# 		target_batches.append(target_data[i:i + batch_size])

# 	return input_batches, target_batches


# if __name__ == '__main__':
# 	DIR = '../../data/processed/'
# 	data = h5py.File('%sdata%d' % (DIR, 5))['data'].value

# 	combined, clean, noise = zip(data)
# 	combined = combined[0]
# 	clean = clean[0]
# 	noise = noise[0]
# 	target = np.concatenate((clean,noise), axis=2)

# 	combined_batch, target_batch = create_batch(combined, target, 50)

# 	f = h5py.File('%stest_batch' % (DIR), 'w')
# 	f.create_dataset('combined_batch', data=combined_batch[0], compression="gzip", compression_opts=9)
# 	f.create_dataset('target_batch', data=target_batch[0], compression="gzip", compression_opts=9)

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
			for k, c in enumerate(row):
				new_data[i][max_rows_in_sample - num_rows + j][k] = c
	return new_data

if __name__ == '__main__':
	processed_data = []

	noise_data = [wavfile.read(INPUT_NOISE_DIR + noise)[1] for noise in os.listdir(INPUT_NOISE_DIR) if noise[-4:] == '.wav']
	noise_data = noise_data[:5]

	batch_size = 50
	curr = 0
	curr_batch = 0

	for i, clean in enumerate(os.listdir(INPUT_CLEAN_DIR)):
		if i > 800:
			continue
			
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

				settings = Sx_combined.stft_settings
				orig_length = len(Sx_clean)
				settings['orig_length'] = orig_length

				processed_data.append([Sx_combined, Sx_clean, Sx_noise, Sx_combined.stft_settings])
			
			curr_batch += 1
			if curr_batch == batch_size:
				combined, clean, noise, stft_settings = zip(*processed_data)
				stft_settings = list(stft_settings)
				
				combined_padded = pad_data(combined)
				clean_padded = pad_data(clean)
				noise_padded = pad_data(noise)

				processed_data = np.array([combined_padded, clean_padded, noise_padded])

				f = h5py.File('%stest_batch' % (OUTPUT_DIR), 'w')
				f.create_dataset('data', data=processed_data, compression="gzip", compression_opts=9)
				
				with open('%stest_settings.pkl' % (OUTPUT_DIR), 'wb') as f:
					pickle.dump(stft_settings, f, pickle.HIGHEST_PROTOCOL)

				print('Finished processing %d clean slice files' % (i + 1))
				break
			print('Finished processing %d clean slice files' % (i + 1))
	