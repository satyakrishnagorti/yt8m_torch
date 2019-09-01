import torch
import numpy as np
import tensorflow as tf
from utils import utils

from random import shuffle


class VideoIterable:
    def __init__(self, record_files, num_classes, feature_sizes, feature_names, max_frames, segment_labels):
        self.record_files = record_files
        self.total_files = len(record_files)
        self.num_classes = num_classes
        self.feature_sizes = feature_sizes
        self.feature_names = feature_names
        self.max_frames = max_frames
        self.segment_labels = segment_labels

    def get_generator(self):
        for file in self.record_files:
            self.process_file(file)


    def process_file(self, file):
        for example in tf.python_io.tf_record_iterator(file):
            example = tf.train.SequenceExample.FromString(example)
            self.process_example(example)

    def process_example(self, example):
        # example represents one video
        example_data = {}

        if self.segment_labels:
            example_data["segment_labels"] = np.array(list(set(example.context.feature['segment_labels'].int64_list.value)))
            example_data["segment_scores"] = np.array(example.context.feature['segment_scores'].float_list.value)
            example_data["segment_start_times"] = np.array(list(set(example.context.feature['segment_start_times'].int64_list.value)))
            example_data["labels"] = np.array(list(set(example.context.feature['labels'].int64_list.value)))

        else:
            example_data["labels"] = np.array(list(set(example.context.feature['labels'].int64_list.value)))

        n_frames = len(example.feature_lists.feature_list['audio'].feature)

        rgb_frames = []
        audio_frames = []
        # iterate through frames
        for i in range(n_frames):

            if i < self.max_frames:
                rgb_frames.append(np.array(
                    list(example.feature_lists.feature_list['rgb'].feature[i].bytes_list.value[0]), dtype=np.float32))
                audio_frames.append(np.array(
                    list(example.feature_lists.feature_list['audio'].feature[i].bytes_list.value[0]), dtype=np.float32))
            else:
                break

        rgb_frames = np.array(rgb_frames).reshape(-1, self.feature_sizes[0])
        audio_frames = np.array(audio_frames).reshape(-1, self.feature_sizes[1])

        rgb_matrix = utils.dequantize(rgb_frames)
        audio_matrix = utils.dequantize(audio_frames)




class Generators:

    @staticmethod
    def video_generator(record_files):
        raise NotImplementedError


class TFRecordFrameDataSet(torch.utils.data.IterableDataset):

    def __init__(self,
                 record_files,
                 num_classes=3862,
                 feature_sizes=(1024, 128),
                 feature_names=("rgb", "audio"),
                 max_frames=300,
                 segment_labels=False,
                 ):
        super(TFRecordFrameDataSet).__init__()
        self.record_files = shuffle(record_files)
        self.total_files = len(record_files)
        self.num_classes = num_classes
        self.feature_sizes = feature_sizes
        self.feature_names = feature_names
        self.max_frames = max_frames
        self.segment_labels = segment_labels

        assert len(self.feature_names) == len(self.feature_sizes)


    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            raise NotImplementedError