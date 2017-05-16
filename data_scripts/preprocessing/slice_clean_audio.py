from scipy.io import wavfile
import matplotlib.pyplot as plt
import os
import numpy as np

DATA_DIR = '../../data/raw_clean/'
OUTPUT_DIR = '../../data/sliced_clean/'

for f in os.listdir(DATA_DIR):
	if f[-4:] == '.wav':
		rate, data = wavfile.read(DATA_DIR + f)
		clean_samples = 0
		clean_frame_threshold = 1300
		window_size = 10000

		frame_slice_ix = []
		silent_frame = True

		moving_average = np.average(np.absolute(data[0:window_size]))
		skip_i = 0

		for i in xrange(window_size, len(data) - window_size):
			moving_average = moving_average * (1 - 1.0 / window_size) + np.absolute(data[i]) / float(window_size)
			
			if silent_frame:
				if moving_average > clean_frame_threshold:
					silent_frame = False
					frame_slice_ix.append(i)
			else:
				if moving_average < clean_frame_threshold:
					silent_frame = True
					frame_slice_ix.append(i)

		for i in xrange(0, len(frame_slice_ix), 2):
			filename = '%s%s_%d.wav' % (OUTPUT_DIR, f[:-4], i / 2)
			wavfile.write(filename, rate, data[frame_slice_ix[i]:frame_slice_ix[i + 1]])
					
