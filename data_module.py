from os import path
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

import json
import glob
from collections import OrderedDict

import torch
import torchaudio
from torch.utils.data import Dataset
from augment import change_pitch_tempo
import random
import os

class MSSTrainDataset(Dataset):
    def __init__(self, root, metadata_path, num_steps, train_batch_size,
                 seg_len=44100 * 11, shift=44100, sample_rate=44100, channels=2):
        self.root = root
        self.metadata = OrderedDict(json.load(open(metadata_path)))
        self.num_steps = num_steps
        self.train_batch_size = train_batch_size
        self.sources = ["drums.wav", "bass.wav", "other.wav", "vocals.wav"]
        self.seg_len = seg_len
        self.shift = shift
        self.channels = channels
        self.sample_rate = sample_rate
        self.data_idx = []

        # Create index for each track in training data
        for file_name, info in self.metadata.items():
            '''
            Calculate the starting point of each segment
            self.data_idx looks like this:

            [('song1', 0), ('song1', 44100), ('song1', 88200)....('song6', 13230).....]

            The length of self.data_idx is the number of total samples
            '''
            for i in range(0, info["length"] - seg_len + 1, shift):
                self.data_idx.append((file_name, i))

    def __len__(self):
        return self.num_steps * self.train_batch_size * 8 # num_gpu
        # return len(self.data_idx)

    def __getitem__(self, index):
        wavs = []
        # Randomly choose sources from different songs and stack them to get a new segment for data augmentation
        for source in self.sources:
            rand_idx = random.randint(0, len(self.data_idx) - 1)

            file_path = self.root + "/" + self.data_idx[rand_idx][0] + "/" + source
            start_idx = self.data_idx[rand_idx][1]
            # print(file_path)
            # wav = change_pitch_tempo(file_path, start_idx, self.seg_len)  # Augmentation
            wav, _ = torchaudio.load(str(file_path), frame_offset=start_idx, num_frames=self.seg_len)
            wavs.append(wav)
        one_sample = torch.stack(wavs)  # [4, 2, 485100]

        return one_sample
    
class MSSValidationDataset(Dataset):
    def __init__(self, root):
        self.root = root
        all_mixtures_path = []     
        for valid_path in os.listdir(self.root):
            part = sorted(glob.glob(os.path.join(self.root, valid_path) + '/*mixture.wav'))
            if len(part) == 0:
                print('No validation data found in: {}'.format(os.path.join(self.root, valid_path)))
            all_mixtures_path += part

        self.list_of_files = all_mixtures_path
        

    def __len__(self):
        return len(self.list_of_files)

    def __getitem__(self, index):
        wav, _ = torchaudio.load(self.list_of_files[index])
        # print(wav.shape, wav.type)
        return wav, index



class WaveBleedingDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.train_root_dir = config.training.train_root_dir
        self.valid_root_dir = config.training.valid_root_dir
        self.meta_dir = config.training.meta_dir
        self.seg_len = config.training.seg_len
        self.steps = config.training.num_steps
        self.shift = config.training.shift
        self.train_batch_size = config.training.train_batch_size
        self.valid_batch_size = config.training.valid_batch_size
        self.meta_train = config.training.meta_train

    def setup(self, stage: str = None) -> None:
        self.train_set = MSSTrainDataset(self.train_root_dir,
                                 path.join(self.meta_dir, self.meta_train),
                                 self.steps, self.train_batch_size,
                                 seg_len=self.seg_len,
                                 shift=self.shift)
        # print(len(self.train_set))
        self.valid_set = MSSValidationDataset(self.valid_root_dir)

    def train_dataloader(self):
        return DataLoader(self.train_set, self.train_batch_size, shuffle=True, num_workers=64, drop_last=True,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.valid_set, self.valid_batch_size, shuffle=False, num_workers=64, drop_last=False,
                          pin_memory=True)

