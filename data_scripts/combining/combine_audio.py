from scipy.io import wavfile
import os
import numpy as np

INPUT_NOISE_DIR = '../../data/raw_noise/'
INPUT_CLEAN_DIR = '../../data/sliced_clean/'
OUTPUT_DIR = '../../data/combined/'

for clean in os.listdir(INPUT_CLEAN_DIR):
	for noise in os.listdir(INPUT_NOISE_DIR):
		if clean[-4:] == '.wav' and noise[-4:] == '.wav':
			rate_clean, data_clean = wavfile.read(INPUT_CLEAN_DIR + clean)
			rate_noise, data_noise = wavfile.read(INPUT_NOISE_DIR + noise)

			length = len(data_clean)
			print('data_clean: %s' % (data_clean))

			data_noise = data_noise[:length]

			average = [(s1/2 + s2/2) for (s1, s2) in zip(data_clean, data_noise)]

			filename = '%s%s.wav' % (OUTPUT_DIR, clean[:-4])
			
			wavfile.write(filename, rate_clean, np.asarray(average, dtype=np.int16))
