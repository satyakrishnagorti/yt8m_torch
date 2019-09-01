import sys
import glob
import pickle
import numpy as np
import tensorflow as tf

from tqdm import tqdm


def create_vid2file_index(files):
    vid2file = {}
    for fname in tqdm(files):
        for example in tf.python_io.tf_record_iterator(fname):
            example = tf.train.SequenceExample.FromString(example)
            vid = example.context.feature['id'].bytes_list.value[0].decode(encoding='UTF-8')
            vid2file[vid] = fname
    return vid2file


def dump_pickle(data, fname):
    with open(fname, 'wb') as f:
        pickle.dump(data, f)


def create_and_dump(all_files_and_names):
    for files, fname in all_files_and_names:
        index = create_vid2file_index(files)
        dump_pickle(index, fname)


if __name__ == '__main__':

    v2_train_files = glob.glob("/data2/yt8m/v2/frame/train*.tfrecord")
    v2_validate_files = glob.glob("/data2/yt8m/v2/frame/validate*.tfrecord")
    v3_validate_files = glob.glob("/data/yt8m/v3/frame/validate*.tfrecord")
    v3_test_files = glob.glob("/data/yt8m/v3/frame/test*.tfrecord")
    v2_train_validate_files = v2_train_files + v2_validate_files

    all_files_and_names = [
        (v2_train_files, '/home/satya/Documents/workspace/yt8m_torch/data/v2_train_files.pkl'),
        (v2_validate_files, '/home/satya/Documents/workspace/yt8m_torch/data/v2_validate_files.pkl'),
        (v2_train_validate_files, '/home/satya/Documents/workspace/yt8m_torch/data/v2_train_validate_files.pkl'),
        (v3_validate_files, '/home/satya/Documents/workspace/yt8m_torch/data/v3_validate_files.pkl'),
        (v3_test_files, '/home/satya/Documents/workspace/yt8m_torch/data/v3_test_files.pkl'),
    ]

    create_and_dump(all_files_and_names)
