### Dataset types for training

* **MUSDB18HQ**: Each different folder represents a different song. There are a total of **100** songs. Each folder contains all needed stems in format `stem name.wav` and the sum of all stems for song - `mixture.wav`.  In latest code releases it's possible to use `flac` instead of `wav`. 

Example:
```
--- Song 1:
------ vocals.wav  
------ bass.wav 
------ drums.wav
------ other.wav
------ mixture.wav
--- Song 2:
------ vocals.wav  
------ bass.wav 
------ drums.wav
------ other.wav
------ mixture.wav
--- Song 3:
...........
```

### Dataset for validation

* **MUSDB18HQ**: The structure of the validation set is the same as training set, except that it contains **50** songs.

Example:
```
--- Song 1:
------ vocals.wav  
------ bass.wav 
------ drums.wav
------ other.wav
------ mixture.wav
--- Song 2:
------ vocals.wav  
------ bass.wav 
------ drums.wav
------ other.wav
------ mixture.wav
--- Song 3:
...........
```