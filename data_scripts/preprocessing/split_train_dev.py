import sys
sys.path.append('../../')
import os
import numpy as np
from config import Config
from create_wavefile import *
import random
import hickle as hkl

DIR = '../../data/processed/'

def create_batch(input_data, target_data, batch_size):
	input_batches = []
	target_batches = []
	
	for i in xrange(0, len(target_data), batch_size):
		input_batches.append(input_data[i:i + batch_size])
		target_batches.append(target_data[i:i + batch_size])
	
	return input_batches, target_batches

def pad_batches(input_batches):
	padded_batches = []
	for i in xrange(len(input_batches)):
		batch = input_batches[i]
		max_len = max(len(s[0]) for s in batch)

		padded_batch = []
		for s in batch:
			if max_len - len(s) > 0:
				padded_batch.append(np.pad(s, ((0, 0),(0, max_len - len(s[0]))), 'constant'))
			else:
				padded_batch.append(s)
		padded_batches.append(padded_batch)

	return padded_batches

if __name__ == '__main__':

	processed_data = np.load(DIR + 'data.npy')
	print('finished loading data')
	num_data = len(processed_data)
	
	dev_ix = set(random.sample(xrange(num_data), num_data / 5))

	train_data = [s for i, s in enumerate(processed_data) if i not in dev_ix]
	dev_data = [s for i, s in enumerate(processed_data) if i in dev_ix]

	train_input, train_clean, train_noise = zip(*train_data)
	dev_input, dev_clean, dev_noise = zip(*dev_data)

	train_target = np.concatenate((train_clean, train_noise), axis=0)
	dev_target = np.concatenate((dev_clean, dev_noise), axis=0)

	# train_input_batches, train_target_batches = create_batch(train_input, train_target, Config.batch_size)
	# dev_input_batches, dev_target_batches = create_batch(dev_input, dev_target, Config.batch_size)

	# train_padded_input_batches = pad_batches(train_input_batches)
	# dev_padded_input_batches = pad_batches(dev_input_batches)
	# train_padded_target_batches = pad_batches()

	# print(np.array(train_input_batches).shape)

	# np.save(OUTPUT_DIR + 'train_input_batch', train_padded_input_batches)
	# np.save(OUTPUT_DIR + 'train_target_batch', train_target_batches)
	# np.save(OUTPUT_DIR + 'dev_input_batch', dev_padded_input_batches)
	# np.save(OUTPUT_DIR + 'dev_target_batch', dev_target_batches)

	np.save(DIR + 'train_input_batch', train_input)
	np.save(DIR + 'train_target_batch', train_target)
	np.save(DIR + 'dev_input_batch', dev_input)
	np.save(DIR + 'dev_target_batch', dev_target)

