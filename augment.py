import os
import random
import tempfile
import wave
from typing import Optional

import torch
import torchaudio
from torch import nn


class FlipChannels(nn.Module):
    """
    Flip left-right channels, borrowed from Demucs.
    """

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        batch, sources, channels, time = wav.size()
        left = torch.randint(2, (batch, sources, 1, 1), device=wav.device)
        left = left.expand(-1, -1, -1, time)
        right = 1 - left
        wav = torch.cat([wav.gather(2, left), wav.gather(2, right)], dim=2)
        return wav


class FlipSign(nn.Module):
    """
    Random sign flip (positive or negative), borrowed from Demucs.
    """

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        batch, sources, channels, time = wav.size()
        signs = torch.randint(2, (batch, sources, 1, 1), device=wav.device, dtype=torch.float32)
        wav = wav * (2 * signs - 1)
        return wav


class Shift(nn.Module):
    """
    Randomly shift audio in time by up to `shift` samples, borrowed from Demucs.
    """

    def __init__(self, shift: int = 44100) -> None:
        super().__init__()
        self.shift = shift

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        batch, sources, channels, time = wav.size()
        length = time - self.shift
        offsets = torch.randint(self.shift, [batch, sources, 1, 1], device=wav.device)
        offsets = offsets.expand(-1, sources, channels, -1)
        indexes = torch.arange(length, device=wav.device)
        wav = wav.gather(3, indexes + offsets)
        return wav


class Scale(nn.Module):
    """
    Randomly scaling the amplitude of audio based on probability
    """
    def __init__(self, proba: float = 1.0, low: float = 0.25, high: float = 1.25) -> None:
        super().__init__()
        self.proba = proba
        self.low = low
        self.high = high

    def forward(self, wav):
        batch, sources, channels, time = wav.size()
        device = wav.device
        if random.random() < self.proba:
            scales = torch.empty(batch, sources, 1, 1, device=device).uniform_(self.low, self.high)
            wav *= scales
        return wav


class Remix(nn.Module):
    """
    Randomly Remix within batch with different sources. It works like below:

    Loop 1, permute source 1:
    [0, 0] <== [3, 0] -------> [batch, source]
    [1, 0] <== [2, 0]
    [2, 0] <== [0, 0]
    [3, 0] <== [1, 0]
    Loop 2, permute source 2:
    [0, 1] <== [2, 1]
    [1, 1] <== [1, 1]
    [2, 1] <== [0, 1]
    [3, 1] <== [3, 1]

    In each loop, it assigns a type of source content from different batch

    For example, finally you will have [[3, 0], [2, 1]] for song 1 instead of [[0, 0], [0, 1]].
    Among them, source 0 comes from Song 3 and source 1 comes from Song 2.
    (assuming each song only has two sources)
    """

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        batch_size, sources, channels, time = wav.size()
        mixed_audio = torch.zeros_like(wav)

        for source in range(sources):
            perm_batch = torch.randperm(batch_size)
            for idx in range(batch_size):
                mixed_audio[idx, source] = wav[perm_batch[idx], source]

        return mixed_audio


def change_pitch_tempo(file_path: str, start_idx: int, seg_len: int,
                       proba: float = 0.2, max_pitch: int = 6, max_tempo: int = 12, 
                       tempo_std: float = 5.0) -> Optional[torch.Tensor]:
    """
    We choose RubberBand command tool to repitch or change the tempo of audio file.
    See https://breakfastquay.com/rubberband/ for more details.

    Step 1: Read the raw audio clip
    Step 2: Save and create a new temp WAV file
    Step 3: Use RubberBand to do pitch/tempo change
    Step 4: Return the new WAV tensor
    """

    with wave.open(file_path, 'rb') as wav_file:
        # Get basic info
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()

        # Seek to the start position in the file
        wav_file.setpos(start_idx)

        # Read the audio data for the specific length by using the pointer we get above
        raw_data = wav_file.readframes(seg_len)

        infile = tempfile.NamedTemporaryFile(suffix=".wav")
        outfile = tempfile.NamedTemporaryFile(suffix=".wav")

        with wave.open(infile.name, 'wb') as temp_wav:
            # Set the parameters for the new temp file
            temp_wav.setnchannels(num_channels)
            temp_wav.setsampwidth(sample_width)
            temp_wav.setframerate(sample_rate)
            temp_wav.writeframes(raw_data)

        out_length = int((1 - 0.01 * max_tempo) * seg_len)

        if random.random() < proba:
            delta_pitch = random.randint(-max_pitch, max_pitch)
            delta_tempo = random.gauss(0, tempo_std)
            delta_tempo = max(min(max_tempo, delta_tempo), -max_tempo)
            delta_tempo = (1 + delta_tempo / 100.0)

            # Set Command
            command = [
                "rubberband-r3",
                f"--pitch {delta_pitch}",
                f"--tempo {delta_tempo:.6f}",
                # f"--pitch-hq",
                # f"--fine",
                f"--fast",
                f"--quiet",
                infile.name,
                outfile.name,
                "> /dev/null 2>&1"
            ]

            command = ' '.join(command)
            # print(command)
            os.system(command)

            new_wav, _ = torchaudio.load(outfile.name)
            new_wav = new_wav[..., :out_length]

            return new_wav

        else:
            new_wav, _ = torchaudio.load(infile.name)
            new_wav = new_wav[..., :out_length]
            return new_wav

