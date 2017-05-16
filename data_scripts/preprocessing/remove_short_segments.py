from scipy.io import wavfile
import matplotlib.pyplot as plt
import os
import numpy as np

DATA_DIR = '../../data/sliced_clean/'

for f in os.listdir(DATA_DIR):
	if f[-4:] == '.wav':
		rate, data = wavfile.read(DATA_DIR + f)
		file_length = len(data) / float(rate)
		if file_length < 1:
			os.remove(DATA_DIR + f)
			print 'removed file %s which had length %f seconds' % (DATA_DIR + f, file_length)