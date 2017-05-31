import time
import argparse
import math
import random
import os
# uncomment this line to suppress Tensorflow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from six.moves import xrange as range

from utils import *
import pdb
from time import gmtime, strftime

from config import Config
from model import SeparationModel
import h5py


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

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--train_path', nargs='?', default='./data/hw3_train.dat', type=str,
    #                     help="Give path to training data - this should not need to be changed if you are running from the assignment directory")
    # parser.add_argument('--val_path', nargs='?', default='./data/hw3_val.dat', type=str,
    #                     help="Give path to val data - this should not need to be changed if you are running from the assignment directory")
    # parser.add_argument('--save_every', nargs='?', default=None, type=int,
    #                     help="Save model every x iterations. Default is not saving at all.")
    # parser.add_argument('--print_every', nargs='?', default=10, type=int,
    #                     help="Print some training and val examples (true and predicted sequences) every x iterations. Default is 10")
    # parser.add_argument('--save_to_file', nargs='?', default='saved_models/saved_model_epoch', type=str,
    #                     help="Provide filename prefix for saving intermediate models")
    # parser.add_argument('--load_from_file', nargs='?', default=None, type=str,
    #                     help="Provide filename to load saved model")
    # args = parser.parse_args()

    logs_path = "tensorboard/" + strftime("%Y_%m_%d_%H_%M_%S", gmtime())

    DIR = 'data/processed/'
    TESTING_MODE = True

    # train_input_batch_name = 'train_input_batch.npy'
    # train_target_batch_name = 'train_target_batch.npy'
    # dev_input_batch_name = 'dev_input_batch.npy'
    # dev_target_batch_name = 'dev_target_batch.npy'

    # if TESTING_MODE:
    #     train_input_batch_name = 'smaller_' + train_input_batch_name
    #     train_target_batch_name = 'smaller_' + train_target_batch_name
    #     dev_input_batch_name = 'smaller_' + dev_input_batch_name
    #     dev_target_batch_name = 'smaller_' + dev_target_batch_name

    data = h5py.File('%sdata%d' % (DIR, 0))['data'].value
    np.append(data, h5py.File('%sdata%d' % (DIR, 1))['data'].value)

    combined, clean, noise = zip(data)
    combined = combined[0]
    clean = clean[0]
    noise = noise[0]
    
    target = np.concatenate((clean,noise), axis=2)

    num_data = len(combined)
    dev_ix = set(random.sample(xrange(num_data), num_data / 5))

    train_input = [s for i, s in enumerate(combined) if i not in dev_ix]
    train_target = [s for i, s in enumerate(target) if i not in dev_ix]
    dev_input = [s for i, s in enumerate(combined) if i in dev_ix]
    dev_target = [s for i, s in enumerate(target) if i in dev_ix]

    # data = np.load(DIR + 'data0.npz')['arr_0']
    # train_input, train_clean, train_noise = zip(train_data)
    # train_target = np.concatenate((train_clean,train_noise), axis=3)
    # dev_input, dev_clean, dev_noise = zip(dev_data)
    # dev_target = np.concatenate((dev_clean,dev_noise), axis=3)

    train_input_batch, train_target_batch = create_batch(train_input, train_target, Config.batch_size)
    dev_input_batch, dev_target_batch = create_batch(dev_input, dev_target, Config.batch_size)
    
    # train_input = clean_data(train_input)
    # train_target = clean_data(train_target)
    # dev_input = clean_data(dev_input)
    # dev_target = clean_data(dev_target)
    # train_input = np.asarray((np.asarray((np.asarray((np.asarray((i for i in r), dtype=np.float64) for r in sample), dtype=np.float64) for sample in batch), dtype=np.float64) for batch in train_input), dtype=np.float64)
    # train_target = np.asarray((np.asarray((np.asarray((np.asarray((i for i in r), dtype=np.float64) for r in sample), dtype=np.float64) for sample in batch), dtype=np.float64) for batch in train_target), dtype=np.float64)
    # dev_input = np.asarray((np.asarray((np.asarray((np.asarray((i for i in r), dtype=np.float64) for r in sample), dtype=np.float64) for sample in batch), dtype=np.float64) for batch in dev_input), dtype=np.float64)
    # dev_target = np.asarray((np.asarray((np.asarray((np.asarray((i for i in r), dtype=np.float64) for r in sample), dtype=np.float64) for sample in batch), dtype=np.float64) for batch in dev_target), dtype=np.float64)

    # print(train_input.shape)

    # def pad_all_batches(batch_feature_array):
    #     for batch_num in range(len(batch_feature_array)):
    #         batch_feature_array[batch_num] = pad_sequences(batch_feature_array[batch_num])[0]
    #     return batch_feature_array

    # train_feature_minibatches = pad_all_batches(train_feature_minibatches)
    # val_feature_minibatches = pad_all_batches(val_feature_minibatches)

    # num_examples = np.sum([train_input.shape[0] for batch in train_feature_minibatches])
    # num_batches_per_epoch = int(math.ceil(num_examples / Config.batch_size))

    num_data = np.sum(len(batch) for batch in train_input_batch)
    num_batches_per_epoch = int(math.ceil(num_data / Config.batch_size))
    num_dev_data = np.sum(len(batch) for batch in dev_input_batch)
    num_dev_batches_per_epoch = int(math.ceil(num_dev_data / Config.batch_size))

    with tf.Graph().as_default():
        model = SeparationModel()
        init = tf.global_variables_initializer()

        saver = tf.train.Saver(tf.trainable_variables())

        with tf.Session() as session:
            # Initializate the weights and biases
            session.run(init)
            # if args.load_from_file is not None:
            #     new_saver = tf.train.import_meta_graph('%s.meta' % args.load_from_file, clear_devices=True)
            #     new_saver.restore(session, args.load_from_file)

            train_writer = tf.summary.FileWriter(logs_path + '/train', session.graph)

            global_start = time.time()

            step_ii = 0

            for curr_epoch in range(Config.num_epochs):
                total_train_cost = total_train_wer = 0
                start = time.time()

                for batch in random.sample(range(num_batches_per_epoch), num_batches_per_epoch):
                    cur_batch_size = len(train_target_batch[batch])

                    batch_cost, summary = model.train_on_batch(session, 
                                                            train_input_batch[batch],
                                                            train_target_batch[batch], 
                                                            train=True)
                    # print('logits: %s' % (logits))
                    total_train_cost += batch_cost * cur_batch_size
                    train_writer.add_summary(summary, step_ii)

                    step_ii += 1
                # input_cost, summary = model.train_on_batch(session, train_input_batch, train_target_batch, train=True)
                # total_train_cost += input_cost

                # train_writer.add_summary(summary, step_ii)
                # step_ii += 1

                train_cost = total_train_cost

                num_dev_batches = len(dev_target_batch)
                total_batch_cost = 0
                total_batch_examples = 0

                # val_batch_cost, _ = model.train_on_batch(session, dev_input_batch[0], dev_target_batch[0], train=False)
                for batch in random.sample(range(num_dev_batches_per_epoch), num_dev_batches_per_epoch):
                    cur_batch_size = len(dev_target_batch[batch])
                    total_batch_examples += cur_batch_size

                    _val_batch_cost, _ = model.train_on_batch(session, dev_input_batch[batch], dev_target_batch[batch], train=False)

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

                # if args.save_every is not None and args.save_to_file is not None and (
                #     curr_epoch + 1) % args.save_every == 0:
                #     saver.save(session, args.save_to_file, global_step=curr_epoch + 1)



