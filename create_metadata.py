import os
import torchaudio
import tqdm
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple, List

# Read dataset.md in case you don't understand the operation of this code

def create_metadata(root_dir: str, usage: str = "train") -> None:

    """
    Creates metadata for all tracks in a given directory and writes it to a JSON file.

    Args:
        root_dir (str): The root directory containing song folders.
        usage (str): The usage type of the dataset, e.g., "train" or "valid". Defaults to "train".
    """

    temp: List[Tuple[str, ThreadPoolExecutor]] = []
    meta_data: Dict[str, Dict[str, str]] = {}

    with ThreadPoolExecutor(16) as pool:
        for subdir, dirs, files in os.walk(root_dir):
            # Skip hidden directories and the root directory itself
            if subdir.startswith('.') or dirs or subdir == root_dir:
                continue

            song_name = str(subdir.split("/")[-1])
            temp.append((song_name, pool.submit(_track_meta, subdir)))

        for name, info in tqdm.tqdm(temp, ncols=128):
            meta_data[name] = info.result()

    json.dump(meta_data, open(f"/23SA01/codes/Music-Source-Separation-BSRoFormer-pl/dataset_meta/data_bleeding_{usage}.json", "w"))


def _track_meta(song_path: str) -> Dict[str, int]:

    """
    Extract metadata from audio files in a given song directory.

    Args:
        song_path (str): Path to the song directory containing audio files.

    Returns:
        Dict[str, int]: A dictionary containing the length and samplerate of the track.
    """

    sources = ['drums.wav', 'vocals.wav', 'other.wav', 'bass.wav']
    track_length = None
    track_samplerate = None

    for source in sources:
        file_path = song_path + "/" + source
        try:
            info = torchaudio.info(str(file_path))
        except RuntimeError:
            print(file_path)
            raise
        length = info.num_frames
        if track_length is None:
            # Set initial values for length and sample rate
            track_length = length
            track_samplerate = info.sample_rate
        elif track_length != length:
            raise ValueError(
                f"Invalid length for file {file_path}: "
                f"expecting {track_length} but got {length}.")
        elif info.sample_rate != track_samplerate:
            raise ValueError(
                f"Invalid sample rate for file {file_path}: "
                f"expecting {track_samplerate} but got {info.sample_rate}.")

    return {"length": length, "samplerate": track_samplerate}


if __name__ == "__main__":

    # Replace the path with the path of your training and validaion sets

    train_dataset_root = '/23SA01/datasets/musdb18hq/train'
    create_metadata(train_dataset_root, usage="train")

    valid_dataset_root = '/23SA01/datasets/musdb18hq/test'
    create_metadata(valid_dataset_root, usage="valid")





