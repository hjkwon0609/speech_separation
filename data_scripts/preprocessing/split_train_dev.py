import sys
sys.path.append('../../')
import os
import numpy as np
from config import Config
import random

DIR = '../../data/processed/'

def create_batch(input_data, target_data, batch_size):
	input_batches = []
	target_batches = []
	
	for i in xrange(0, len(target_data), batch_size):
		input_batches.append(input_data[i:i + batch_size])
		target_batches.append(target_data[i:i + batch_size])
	
	return input_batches, target_batches

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

	num_data = len(processed_data)
	
	dev_ix = set(random.sample(xrange(num_data), num_data / 5))

	train_data = [s for i, s in enumerate(total_data) if i not in dev_ix]
	dev_data = [s for i, s in enumerate(total_data) if i in dev_ix]

	train_input, train_target = zip(*train_data)
	dev_input, dev_target = zip(*dev_data)

	train_input_batches, train_target_batches = create_batch(train_input, train_target, Config.batch_size)
	dev_input_batches, dev_target_batches = create_batch(dev_input, dev_target, Config.batch_size)

