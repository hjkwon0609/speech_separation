import time
import argparse
import math
import random
import os
import distutils.util
# uncomment this line to suppress Tensorflow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from six.moves import xrange as range
from scipy.io import wavfile

from utils import *
import pdb
from time import gmtime, strftime

from config import Config
from model import SeparationModel
import h5py
from stft.types import SpectrogramArray
import stft


def clean_data(data):
    # hack for now so that I don't have to preprocess again
    num_batches = len(data)
    print(num_batches)
    num_samples_in_batch = len(data[0])
    print(num_samples_in_batch)
    num_rows_in_sample = len(data[0][0])
    print(num_rows_in_sample)
    num_cols_in_row = len(data[0][0][0])
    print(num_cols_in_row)
    new_data = np.zeros((num_batches, num_samples_in_batch, num_rows_in_sample, num_cols_in_row))
    for i, batch in enumerate(data):
        for j, sample in enumerate(batch):
            for k, r in enumerate(sample):
                for l, c in enumerate(r):
                    new_data[i][j][k][l]
    return new_data

def create_batch(input_data, target_data, batch_size):
    input_batches = []
    target_batches = []
    
    for i in xrange(0, len(target_data), batch_size):
        input_batches.append(input_data[i:i + batch_size])
        target_batches.append(target_data[i:i + batch_size])
    
    return input_batches, target_batches

def model_train(freq_weighted):
    logs_path = "tensorboard/" + strftime("%Y_%m_%d_%H_%M_%S", gmtime())

    DIR = 'data/processed/'
    TESTING_MODE = True

    data = h5py.File('%sdata%d' % (DIR, 0))['data'].value
    np.append(data, h5py.File('%sdata%d' % (DIR, 1))['data'].value)

    combined, clean, noise = zip(data)
    combined = combined[0]
    clean = clean[0]
    noise = noise[0]
    
    target = np.concatenate((clean,noise), axis=2)

    num_data = len(combined)
    random.seed(1)
    dev_ix = set(random.sample(xrange(num_data), num_data / 5))

    train_input = [s for i, s in enumerate(combined) if i not in dev_ix]
    train_target = [s for i, s in enumerate(target) if i not in dev_ix]
    dev_input = [s for i, s in enumerate(combined) if i in dev_ix]
    dev_target = [s for i, s in enumerate(target) if i in dev_ix]

    train_input_batch, train_target_batch = create_batch(train_input, train_target, Config.batch_size)
    dev_input_batch, dev_target_batch = create_batch(dev_input, dev_target, Config.batch_size)

    num_data = np.sum(len(batch) for batch in train_input_batch)
    num_batches_per_epoch = int(math.ceil(num_data / Config.batch_size))
    num_dev_data = np.sum(len(batch) for batch in dev_input_batch)
    num_dev_batches_per_epoch = int(math.ceil(num_dev_data / Config.batch_size))

    with tf.Graph().as_default():
        model = SeparationModel(freq_weighted=freq_weighted)
        init = tf.global_variables_initializer()

        saver = tf.train.Saver(tf.trainable_variables())

        with tf.Session() as session:
            session.run(init)
            
            # if args.load_from_file is not None:
            #     new_saver = tf.train.import_meta_graph('%s.meta' % args.load_from_file, clear_devices=True)
            #     new_saver.restore(session, args.load_from_file)

            train_writer = tf.summary.FileWriter(logs_path + '/train', session.graph)

            global_start = time.time()

            step_ii = 0

            for curr_epoch in range(Config.num_epochs):
                total_train_cost = 0
                total_train_examples = 0

                start = time.time()

                for batch in random.sample(range(num_batches_per_epoch), num_batches_per_epoch):
                    cur_batch_size = len(train_target_batch[batch])
                    total_train_examples += cur_batch_size

                    _, batch_cost, summary = model.train_on_batch(session, 
                                                            train_input_batch[batch],
                                                            train_target_batch[batch], 
                                                            train=True)

                    total_train_cost += batch_cost * cur_batch_size
                    train_writer.add_summary(summary, step_ii)

                    step_ii += 1

                train_cost = total_train_cost / total_train_examples

                num_dev_batches = len(dev_target_batch)
                total_batch_cost = 0
                total_batch_examples = 0

                # val_batch_cost, _ = model.train_on_batch(session, dev_input_batch[0], dev_target_batch[0], train=False)
                for batch in random.sample(range(num_dev_batches_per_epoch), num_dev_batches_per_epoch):
                    cur_batch_size = len(dev_target_batch[batch])
                    total_batch_examples += cur_batch_size

                    _, _val_batch_cost, _ = model.train_on_batch(session, dev_input_batch[batch], dev_target_batch[batch], train=False)

                    total_batch_cost += cur_batch_size * _val_batch_cost


                val_batch_cost = None
                try:
                    val_batch_cost = total_batch_cost / total_batch_examples
                except ZeroDivisionError:
                    val_batch_cost = 0

                log = "Epoch {}/{}, train_cost = {:.3f}, val_cost = {:.3f}, time = {:.3f}"
                print(
                log.format(curr_epoch + 1, Config.num_epochs, train_cost, val_batch_cost, time.time() - start))

                # if args.print_every is not None and (curr_epoch + 1) % args.print_every == 0:
                #     batch_ii = 0
                #     model.print_results(train_feature_minibatches[batch_ii], train_labels_minibatches[batch_ii])

                if (curr_epoch + 1) % 10 == 0:
                    checkpoint_name = 'checkpoints/%dlayer_%flr_model' % (Config.num_layers, Config.lr)
                    if freq_weighted:
                        checkpoint_name = checkpoint_name + '_freq_weighted'
                    saver.save(session, checkpoint_name, global_step=curr_epoch + 1)


def model_test(test_input):

    test_rate, test_audio = wavfile.read(test_input)
    test_spec = stft.spectrogram(test_audio)

    test_data = np.array([test_spec.transpose() / 100000])  # make data a batch of 1

    with tf.Graph().as_default():
        model = SeparationModel()
        saver = tf.train.Saver(tf.trainable_variables())
        
        with tf.Session() as session:
            ckpt = tf.train.get_checkpoint_state('checkpoints/')
            if ckpt: #and tf.gfile.Exists(ckpt.model_checkpoint_path):
                print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                saver.restore(session, ckpt.model_checkpoint_path)
            else:
                print("Created model with fresh parameters.")
                session.run(tf.initialize_all_variables())

            test_data_shape = np.shape(test_data)
            dummy_target = np.zeros((test_data_shape[0], test_data_shape[1], 2 * test_data_shape[2]))

            output, _, _ = model.train_on_batch(session, test_data, dummy_target, train=False)

            num_freq_bin = output.shape[2] / 2
            clean_output = output[0,:,:num_freq_bin]
            noise_output = output[0,:,num_freq_bin:]

            clean_mask, noise_mask = create_mask(clean_output, noise_output)

            clean_spec = createSpectrogram(np.multiply(clean_mask.transpose(), test_spec), test_spec) 
            noise_spec = createSpectrogram(np.multiply(noise_mask.transpose(), test_spec), test_spec)

            clean_wav = stft.ispectrogram(clean_spec)
            noise_wav = stft.ispectrogram(noise_spec)

            writeWav('data/test_combined/output_clean.wav', 44100, clean_wav)
            writeWav('data/test_combined/output_noise.wav', 44100, noise_wav)

def writeWav(fn, fs, data):
    data = data * 1.5 / np.max(np.abs(data))
    wavfile.write(fn, fs, data)


def create_mask(clean_output, noise_output, hard=True):
    clean_mask = np.zeros(clean_output.shape)
    noise_mask = np.zeros(noise_output.shape)

    if hard:
        for i in range(len(clean_output)):
            for j in range(len(clean_output[0])):
                if abs(clean_output[i][j]) < abs(noise_output[i][j]):
                    noise_mask[i][j] = 1.0
                else:
                    clean_mask[i][j] = 1.0
    else:
        for i in range(len(clean_output)):
            for j in range(len(clean_output[0])):
                clean_mask[i][j] = abs(clean_output[i][j]) / (abs(clean_output[i][j]) + abs(noise_output[i][j]))
                noise_mask[i][j] = abs(noise_output[i][j]) / (abs(clean_output[i][j]) + abs(noise_output[i][j]))


    return clean_mask, noise_mask

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', nargs='?', default=True, type=distutils.util.strtobool)
    parser.add_argument('--test_input', nargs='?', default='data/test_combined/combined.wav', type=str)
    parser.add_argument('--freq_weighted', nargs='?', default=True, type=distutils.util.strtobool)
    args = parser.parse_args()

    if args.train:
        model_train(args.freq_weighted)
    else:
        model_test(args.test_input)

