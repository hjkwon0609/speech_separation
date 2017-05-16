from scipy.io import wavfile
import os
import numpy as np

INPUT_NOISE_DIR = '../../data/raw_noise/'
INPUT_CLEAN_DIR = '../../data/sliced_clean/'
OUTPUT_DIR = '../../data/combined/'

#convert samples from strings to ints
# def bin_to_int(bin):
#     as_int = 0
#     for char in bin[::-1]: #iterate over each char in reverse (because little-endian)
#         #get the integer value of char and assign to the lowest byte of as_int, shifting the rest up
#         as_int <<= 8
#         as_int += ord(char) 
#     return as_int

# def int_to_bin(int_list):
# 	as_bin = ""
# 	print(int_list)
# 	for num in int_list[::-1]:
# 		as_bin += chr(num)
# 		return as_bin

for clean in os.listdir(INPUT_CLEAN_DIR):
	for noise in os.listdir(INPUT_NOISE_DIR):
		if clean[-4:] == '.wav' and noise[-4:] == '.wav':
			rate_clean, data_clean = wavfile.read(INPUT_CLEAN_DIR + clean)
			rate_noise, data_noise = wavfile.read(INPUT_NOISE_DIR + noise)

			print("-----rate_clean------")
			print(rate_clean)
			print("-----rate_noise------")
			print(rate_noise)

			length = len(data_clean)
			print("length: " + str(length))
			data_noise = data_noise[:length]

			average = [(s1/2)+(s2/2) for (s1, s2) in zip(data_clean, data_noise)]
			print("-----data_clean------")
			print(data_clean[:100])
			print("-----data_noise------")
			print(data_noise[:100])
			print("-----ave------")
			print(average[:100])

			filename = '%s%s.wav' % (OUTPUT_DIR, clean[:-4])
			
			wavfile.write(filename, rate_clean, np.array(average))
			assert(False)


# for f in os.listdir(DATA_DIR):
# 	if f[-4:] == '.wav':
# 		rate, data = wavfile.read(DATA_DIR + f)
# 		clean_samples = 0
# 		clean_frame_threshold = 1300
# 		window_size = 10000

# 		frame_slice_ix = []
# 		silent_frame = True

# 		moving_average = np.average(np.absolute(data[0:window_size]))
# 		skip_i = 0

# 		for i in xrange(window_size, len(data) - window_size):
# 			moving_average = moving_average * (1 - 1.0 / window_size) + np.absolute(data[i]) / float(window_size)
			
# 			if silent_frame:
# 				if moving_average > clean_frame_threshold:
# 					silent_frame = False
# 					frame_slice_ix.append(i)
# 			else:
# 				if moving_average < clean_frame_threshold:
# 					silent_frame = True
# 					frame_slice_ix.append(i)

# 		for i in xrange(0, len(frame_slice_ix), 2):
# 			filename = '%s%s_%d.wav' % (OUTPUT_DIR, f[:-4], i / 2)
# 			wavfile.write(filename, rate, data[frame_slice_ix[i]:frame_slice_ix[i + 1]])
					
