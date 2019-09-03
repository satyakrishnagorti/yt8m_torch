import math
import torch
import numpy as np
import tensorflow as tf
from utils import utils

from random import shuffle


class VideoIterable:

    def __init__(self,
                 record_files,
                 num_classes,
                 feature_sizes,
                 feature_names,
                 max_frames,
                 segment_labels,
                 segment_size):
        self.record_files = record_files
        self.total_files = len(record_files)
        self.num_classes = num_classes
        self.feature_sizes = feature_sizes
        self.feature_names = feature_names
        self.max_frames = max_frames
        self.segment_labels = segment_labels
        self.segment_size = segment_size

    def get_generator(self):
        for file in self.record_files:
            for example in tf.python_io.tf_record_iterator(file):
                example = tf.train.SequenceExample.FromString(example)
                yield self.process_example(example)

    def process_example(self, example):
        # example represents one video
        example_data = {}

        if self.segment_labels:
            example_data["segment_indices"] = np.array(list(example.context.feature['segment_labels'].int64_list.value))
            example_data["segment_scores"] = np.array(example.context.feature['segment_scores'].float_list.value)
            example_data["segment_start_times"] = np.array(list(example.context.feature['segment_start_times'].int64_list.value))
            example_data["labels"] = np.array(list(example.context.feature['labels'].int64_list.value))
            example_data["vid"] = example.context.feature['id'].bytes_list.value[0].decode(encoding='UTF-8')

            assert(len(example_data["segment_indices"]) == len(example_data["segment_scores"]))
            assert(len(example_data["segment_indices"]) == len(example_data["segment_start_times"]))
        else:
            example_data["labels"] = np.array(list(set(example.context.feature['labels'].int64_list.value)))
            example_data["vid"] = example.context.feature['id'].bytes_list.value[0].decode(encoding='UTF-8')

        num_frames = len(example.feature_lists.feature_list['audio'].feature)

        rgb_frames = []
        audio_frames = []
        # iterate through frames
        for i in range(num_frames):

            if i < self.max_frames:
                rgb_frames.append(np.array(
                    list(example.feature_lists.feature_list['rgb'].feature[i].bytes_list.value[0]), dtype=np.float32))
                audio_frames.append(np.array(
                    list(example.feature_lists.feature_list['audio'].feature[i].bytes_list.value[0]), dtype=np.float32))
            else:
                break

        rgb_matrix = utils.pad_if_necessary(np.array(rgb_frames).reshape(-1, self.feature_sizes[0]),
                                            self.max_frames)
        audio_matrix = utils.pad_if_necessary(np.array(audio_frames).reshape(-1, self.feature_sizes[1]),
                                              self.max_frames)

        assert(rgb_matrix.shape[0] == self.max_frames)
        assert(audio_matrix.shape[0] == self.max_frames)

        rgb_matrix = utils.dequantize(rgb_matrix)
        audio_matrix = utils.dequantize(audio_matrix)
        video_matrix = np.concatenate((rgb_matrix, audio_matrix), axis=1)
        transformed_data = self.transform_data(video_matrix, example_data)

        return transformed_data

    def transform_data(self, video_matrix, example_data):

        num_frames = video_matrix.shape[0]
        batch_video_matrix = torch.from_numpy(video_matrix)

        if self.segment_labels:
            segment_start_times = example_data["segment_start_times"]
            segment_scores      = example_data["segment_scores"]
            segment_indices     = example_data["segment_indices"]
            video_labels        = example_data["labels"]
            vid                 = example_data["vid"]

            start_times_idxs = segment_start_times.astype(np.float32) / 5
            # array of tuples like, with first value=segment time indexes and second value=class number
            segment_scores = torch.from_numpy(segment_scores)
            # batch_labels dim: (60, num_classes)
            label_indices = torch.from_numpy(np.stack([start_times_idxs, segment_indices], axis=-1)).long()
            batch_labels = torch.sparse.FloatTensor(label_indices.t(), segment_scores,
                                                    torch.Size([60, self.num_classes])).to_dense()
            batch_labels = torch.clamp(batch_labels, 0, 1)

            batch_label_weights = torch.sparse.FloatTensor(label_indices.t(), torch.ones_like(segment_scores),
                                                           torch.Size([60, self.num_classes])).to_dense()
            batch_label_weights = torch.clamp(batch_label_weights, 0, 1)

            # label mask: dim: (60,1)
            label_mask = torch.from_numpy(np.stack([start_times_idxs, np.zeros_like(start_times_idxs)], axis=-1)).long()
            batch_label_masks = torch.sparse.FloatTensor(label_mask.t(),
                                                         torch.ones_like(torch.from_numpy(start_times_idxs)),
                                                         torch.Size([60, 1])).to_dense()
            batch_label_masks = torch.clamp(batch_label_masks, 0, 1)

            total_segments = self.max_frames // self.segment_size
            batch_frames = torch.repeat_interleave(torch.tensor([self.segment_size]), total_segments).view(
                (total_segments, 1))

            # video_labels dim: (num_classes,)
            video_labels = torch.from_numpy(video_labels)
            batch_video_labels = torch.sparse.FloatTensor(video_labels.unsqueeze(0), torch.ones_like(video_labels),
                                     torch.Size([self.num_classes])).to_dense()
            batch_video_labels = torch.clamp(batch_video_labels, 0, 1)

            transformed_data = {
                "video_ids": np.array([vid]),
                "video_matrix": batch_video_matrix,
                "video_labels": batch_video_labels,
                "video_num_frames": torch.tensor([num_frames]),
                "segment_num_frames": batch_frames,
                "segment_labels": batch_labels,
                "label_weights": batch_label_weights,
                "label_masks": batch_label_masks,
            }
        else:
            video_labels = example_data["labels"]
            vid          = example_data["vid"]
            video_labels = torch.from_numpy(video_labels)
            batch_video_labels = torch.sparse.FloatTensor(video_labels.unsqueeze(0), torch.ones_like(video_labels),
                                     torch.Size([self.num_classes])).to_dense()
            batch_video_labels = torch.clamp(batch_video_labels, 0, 1)

            transformed_data = {
                "video_ids": np.array([vid]),
                "video_matrix": batch_video_matrix,
                "video_labels": batch_video_labels,
                "video_num_frames": torch.tensor([num_frames]),
                "segment_num_frames": None,
                "segment_labels": None,
                "label_weights": None,
                "label_masks": None,
            }

        return transformed_data


class TFRecordFrameDataSet(torch.utils.data.IterableDataset):

    def __init__(self,
                 record_files,
                 num_classes=3862,
                 feature_sizes=(1024, 128),
                 feature_names=("rgb", "audio"),
                 max_frames=300,
                 segment_labels=False,
                 segment_size=5
                 ):
        super(TFRecordFrameDataSet).__init__()
        shuffle(record_files)
        self.record_files = record_files
        self.total_files = len(record_files)
        self.num_classes = num_classes
        self.feature_sizes = feature_sizes
        self.feature_names = feature_names
        self.max_frames = max_frames
        self.segment_labels = segment_labels
        self.segment_size = segment_size

        assert len(self.feature_names) == len(self.feature_sizes)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # in a worker process
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            per_worker = int(math.ceil(self.total_files) / float(num_workers))
            file_start = worker_id * per_worker
            file_end = min(file_start + per_worker, self.total_files)
            curr_record_files = self.record_files[file_start: file_end]
        else:
            curr_record_files = self.record_files

        video_generator_obj = VideoIterable(curr_record_files,
                                            self.num_classes,
                                            self.feature_sizes,
                                            self.feature_names,
                                            self.max_frames,
                                            self.segment_labels,
                                            self.segment_size)

        return video_generator_obj.get_generator()

    def get_collate_fn(self):

        segment_labels = self.segment_labels

        def custom_collate_fn(batch_samples):
            # batch_data will be array of transformed_data ^ dicts
            if segment_labels:
                attributes = ["video_ids", "video_matrix", "video_labels",
                              "video_num_frames", "segment_num_frames", "segment_labels",
                              "label_weights", "label_masks"]
            else:
                attributes = ["video_ids", "video_matrix", "video_labels", "video_num_frames"]

            collated_data = {}

            for name in attributes:
                running_list = []
                for item in batch_samples:
                    running_list.append(item[name])
                if name != "video_ids":
                    collated_data[name] = torch.stack(running_list, dim=0)
                else:
                    collated_data[name] = np.stack(running_list, axis=0)

            return collated_data

        return custom_collate_fn


if __name__ == '__main__':
    import glob
    from tqdm import tqdm
    ds = TFRecordFrameDataSet(glob.glob("/data/yt8m/v3/frame/validate*.tfrecord"), segment_labels=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=ds.get_collate_fn(), num_workers=8)
    for elem in tqdm(iter(dl)):
        print(elem)
        break
