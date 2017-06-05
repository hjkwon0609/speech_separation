import sys
import stft
import numpy as np
import numpy.linalg as la
from numpy.linalg import inv
from numpy.linalg import svd
from scipy.io import wavfile
from stft.types import SpectrogramArray
import pickle
import scipy

raw1 = '../../data/sliced_clean/f10_script2_clean_113.wav'
raw2 = "../../data/raw_noise/noise11_1.wav"
merged = '../../data/test_combined/combined.wav'
m_dir = "results/"
separated_dir = "results/"

# ASSUMPTION: len(spec.shape) <= 3
def squeeze(spec):
	if len(spec.shape) > 2:
		spec = np.delete(spec, 1, axis=2)
		spec = spec.squeeze(2)
	return spec

def createSpectrogram(arr, orig):
	x = SpectrogramArray(arr, stft_settings={
								'framelength': orig.stft_settings['framelength'],
								'hopsize': orig.stft_settings['hopsize'],
								'overlap': orig.stft_settings['overlap'],
								'centered': orig.stft_settings['centered'],
								'window': orig.stft_settings['window'],
								'halved': orig.stft_settings['halved'],
								'transform': orig.stft_settings['transform'],
								'padding': orig.stft_settings['padding'],
								'outlength': orig.stft_settings['outlength'],
								}
						)
	return x

def writeWav(fn, fs, data):
		data = data# * 1.5 / np.max(np.abs(data))
		wavfile.write(fn, fs, data)



def createMatrix():
	# spectrogram_arguments = {'framelength': 512, 'overlap': 512, 'window': scipy.signal.hamming(512)}
	def saveFile(fn, data):
		f = open(fn, 'wb')
		pickle.dump(data, f)
		f.close()
	fs1, data1 = wavfile.read(raw1)
	fs2, data2 = wavfile.read(raw2)
	
	minlen = min(len(data1), len(data2))
	data1 = data1[:minlen]
	data2 = data2[:minlen]

	spec1 = stft.spectrogram(data1)
	spec2 = stft.spectrogram(data2)
	
	# Reduce dimension
	spec1 = squeeze(spec1)
	spec2 = squeeze(spec2)

	# same dimensions
	a = np.zeros(spec1.shape)
	b = np.zeros(spec2.shape)

	# hard
	for i in range(len(spec1)):
		for j in range(len(spec1[0])):
			if abs(spec1[i][j]) < abs(spec2[i][j]):
				b[i][j] = 1.0
			else:
				a[i][j] = 1.0

	# soft
	# for i in range(len(spec1)):
	# 	for j in range(len(spec1[0])):
	# 		if (abs(spec1[i][j]) + abs(spec2[i][j])) == 0:
	# 			continue
	# 		a[i][j] = abs(spec1[i][j]) / (abs(spec1[i][j]) + abs(spec2[i][j]))
	# 		b[i][j] = abs(spec2[i][j]) / (abs(spec1[i][j]) + abs(spec2[i][j]))

	fs, data = wavfile.read(merged)
	spec = stft.spectrogram(data)
	spec = squeeze(spec)

	output_a = createSpectrogram(np.multiply(a, spec), spec)
	output_b = createSpectrogram(np.multiply(b, spec), spec)

	output_a2 = stft.ispectrogram(output_a)
	output_b2 = stft.ispectrogram(output_b)

	writeWav(separated_dir + "a.wav", fs1, output_a2)
	writeWav(separated_dir + "b.wav", fs1, output_b2)

	return

	

def divide():
	def loadFile(fn):
		f = open(fn, 'rb')
		data = pickle.load(f)
		f.close()
		return data

	fs, data = wavfile.read(merged)
	spec = stft.spectrogram(data, framelength=512)
	spec = squeeze(spec)
	Ma = loadFile(m_dir + "M_" + raw1[:-4])
	Mb = loadFile(m_dir + "M_" + raw2[:-4])
	a = createSpectrogram(np.dot(Ma, spec), spec)
	b = createSpectrogram(np.dot(Mb, spec), spec)
	
	output_a = stft.ispectrogram(a)
	output_b = stft.ispectrogram(b)

	writeWav(separated_dir + "a.wav", fs, output_a)
	writeWav(separated_dir + "b.wav", fs, output_b)

if __name__ == "__main__":
	# argparse later
	c = sys.argv[1]
	if c == "a":
		createMatrix()
	elif c == "b":
		divide()
	else:
		print "Unknown"

def trash1():
	def getMatrix(s, a, b):
		s_mat = np.zeros([a, b])
		s_mat[:min(a,b), :min(a,b)] = np.diag(s)
		return s_mat

	def getInverseMatrix(s, a, b):
		s = inv(np.diag(s)).diagonal()
		s_mat = np.zeros([b, a])
		s_mat[:min(a,b), :min(a,b)] = np.diag(s)
		return s_mat

	def originalValuesInverse(spec):
		u, s_lin, v = svd(spec)
		s = getInverseMatrix(s_lin, len(u), len(v))
		return inv(u), s, inv(v)

	def newValues(spec):
		u, s_lin, v = svd(spec)
		s = getMatrix(s_lin, len(u), len(v))
		return u, s, v

	ua, sa, va = originalValuesInverse(spec1)
	ub, sb, vb = originalValuesInverse(spec2)

	una, sna, vna = newValues(a)
	unb, snb, vnb = newValues(b)

	# M * A_orig = A_new
	Ma = np.dot(np.dot(np.dot(np.dot(np.dot(una, sna), vna), va), sa), ua)
	Mb = np.dot(np.dot(np.dot(np.dot(np.dot(unb, snb), vnb), vb), sb), ub)

	Ma = createSpectrogram(Ma, spec1)
	Mb = createSpectrogram(Mb, spec2)

	saveFile(m_dir + "M_" + raw1[:-4], Ma)
	saveFile(m_dir + "M_" + raw2[:-4], Mb)

