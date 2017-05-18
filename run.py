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

    train_input = np.load(DIR + 'train_input_batch.npy')
    train_target = np.load(DIR + 'train_target_batch.npy')
    dev_input = np.load(DIR + 'dev_input_batch.npy')
    dev_target = np.load(DIR + 'dev_target_batch.npy')

    # def pad_all_batches(batch_feature_array):
    #     for batch_num in range(len(batch_feature_array)):
    #         batch_feature_array[batch_num] = pad_sequences(batch_feature_array[batch_num])[0]
    #     return batch_feature_array

    # train_feature_minibatches = pad_all_batches(train_feature_minibatches)
    # val_feature_minibatches = pad_all_batches(val_feature_minibatches)

    num_examples = np.sum([batch.shape[0] for batch in train_feature_minibatches])
    num_batches_per_epoch = int(math.ceil(num_examples / Config.batch_size))

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
                    cur_batch_size = len(train_target_minibatches[batch])

                    batch_cost, summary = model.train_on_batch(session, train_feature_minibatches[batch],
                                                               train_target_minibatches[batch], train=True)
                    total_train_cost += batch_cost * cur_batch_size

                    train_writer.add_summary(summary, step_ii)
                    step_ii += 1

                train_cost = total_train_cost / num_examples

                val_batch_cost, _ = model.train_on_batch(session, dev_feature_minibatches[0],
                                                         dev_target_minibatches[0], train=False)

                log = "Epoch {}/{}, train_cost = {:.3f}, train_ed = {:.3f}, val_cost = {:.3f}, val_ed = {:.3f}, time = {:.3f}"
                print(
                log.format(curr_epoch + 1, Config.num_epochs, train_cost, train_wer, val_batch_cost, val_batch_ler,
                           time.time() - start))

                if args.print_every is not None and (curr_epoch + 1) % args.print_every == 0:
                    batch_ii = 0
                    model.print_results(train_feature_minibatches[batch_ii], train_labels_minibatches[batch_ii])

                if args.save_every is not None and args.save_to_file is not None and (
                    curr_epoch + 1) % args.save_every == 0:
                    saver.save(session, args.save_to_file, global_step=curr_epoch + 1)