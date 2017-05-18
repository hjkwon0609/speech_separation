import sys
sys.path.append('../../')
import os
import numpy as np
from config import Config
from create_wavefile import *
import random
import hickle as hkl

DIR = '../../data/processed/'

MAKE_SMALLER = True

def create_batch(input_data, target_data, batch_size):
	input_batches = []
	target_batches = []
	
	for i in xrange(0, len(target_data), batch_size):
		input_batches.append(input_data[i:i + batch_size])
		target_batches.append(target_data[i:i + batch_size])
	
	return input_batches, target_batches

def pad_data(data):
	num_samples = len(data)
	print(num_samples)
	max_rows_in_sample = max(len(data[i]) for i in xrange(num_samples))
	print([len(data[i]) for i in xrange(num_samples)])
	print(max_rows_in_sample)
	num_cols_in_row = data[0][0].size
	print(num_cols_in_row)
	print(data[0][0])
	new_data = np.zeros((num_samples, max_rows_in_sample, num_cols_in_row))
	for i, sample in enumerate(data):
		for j, row in enumerate(sample):
			num_rows = len(row)
			print(row)
			for k, c in enumerate(row):
				new_data[i][max_rows_in_sample - num_rows + j][k] = c
	return new_data
	# padded_batches = 
	# for i in xrange(len(data)):
	# 	batch = data[i]
	# 	max_len = max(s[0].size for s in batch)
	# 	padded_batch = []
	# 	for s in batch:
	# 		if max_len - len(s) > 0:
	# 			padded_batch.append(np.pad(s, ((max_len - len(s[0]), 0),(0, 0)), 'constant'))
	# 		else:
	# 			padded_batch.append(s)
	# 	padded_batches.append(padded_batch)

	# return padded_batches

if __name__ == '__main__':

	processed_data = np.load(DIR + 'data.npy')
	print('finished loading data')
	num_data = len(processed_data)

	###############################################################################
	# preprocess for smaller data to get model working (BEGIN)
	###############################################################################
	if MAKE_SMALLER:
		dev_ix = set(random.sample(xrange(num_data), num_data / 100))
		processed_data = [l for i, l in enumerate(processed_data) if i in dev_ix]
		num_data = len(processed_data)
    ###############################################################################
    # preprocess for smaller data to get model working (END)
    ###############################################################################
    
	dev_ix = set(random.sample(xrange(num_data), num_data / 5))

	processed_data = [np.transpose(s) for s in processed_data]

	inp, clean, noise = zip(*processed_data)
	padded_input = pad_data(inp)
	padded_clean = pad_data(clean)
	padded_noise = pad_data(noise)

	train_padded_input = [s for i, s in enumerate(padded_input) if i not in dev_ix]
	train_padded_clean = [s for i, s in enumerate(padded_clean) if i not in dev_ix]
	train_padded_noise = [s for i, s in enumerate(padded_noise) if i not in dev_ix]
	dev_padded_input = [s for i, s in enumerate(padded_input) if i in dev_ix]
	dev_padded_clean = [s for i, s in enumerate(padded_clean) if i in dev_ix]
	dev_padded_noise = [s for i, s in enumerate(padded_noise) if i in dev_ix]


	# train_input, train_clean, train_noise = zip(*train_data)
	# dev_input, dev_clean, dev_noise = zip(*dev_data)

	# train_padded_input = pad_data(train_input)
	# train_padded_clean = pad_data(train_clean)
	# train_padded_noise = pad_data(train_noise)
	# dev_padded_input = pad_data(dev_input)
	# dev_padded_clean = pad_data(dev_clean)
	# dev_padded_noise = pad_data(dev_noise)

	train_target = np.concatenate((train_padded_clean, train_padded_noise), axis=1)
	dev_target = np.concatenate((dev_padded_clean, dev_padded_noise), axis=1)

	train_input_batches, train_target_batches = create_batch(train_padded_input, train_target, Config.batch_size)
	dev_input_batches, dev_target_batches = create_batch(dev_padded_input, dev_target, Config.batch_size)

	print(np.array(train_input_batches))

	train_input_batch_name = 'train_input_batch'
	train_target_batch_name = 'train_target_batch'
	dev_input_batch_name = 'dev_input_batch'
	dev_target_batch_name = 'dev_target_batch'

	if MAKE_SMALLER:
		train_input_batch_name = 'smaller_' + train_input_batch_name
		train_target_batch_name = 'smaller_' + train_target_batch_name
		dev_input_batch_name = 'smaller_' + dev_input_batch_name
		dev_target_batch_name = 'smaller_' + dev_target_batch_name

	np.save(DIR + train_input_batch_name, train_input_batches)
	np.save(DIR + train_target_batch_name, train_target_batches)
	np.save(DIR + dev_input_batch_name, dev_input_batches)
	np.save(DIR + dev_target_batch_name, dev_target_batches)

