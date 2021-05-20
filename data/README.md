# Dataset Preparation

MusCaps can be trained on datasets of (audio, caption) pairs, organised as follows

```
dataset_name
├── audio            
│   ├── track_1.npy
│   ├── track_2.npy
|   └── ...
├── dataset_train.json    
├── dataset_val.json    
└── dataset_test.json
```

### Captions
Prepare a json file for each data split (`train`, `val`, `test`) with the annotations containing the following fields

```json
{
    "track_id": "track_1", 
    "caption": 
        {
            "raw": "First caption!", 
            "tokens": ["first", "caption"]
        }
}
```
and place it in the `data/datasets/<dataset_name>` folder. A toy example is provided in [`data/datasets/audiocaption`](datasets/audiocaption). To preprocess the raw caption text and obtain the tokens, you'll need to lower case and remove punctuation. You may also want to ensure all captions have a suitable length (e.g. between 3 and 22 tokens).

### Audio
Audio files should be preprocessed and stored as `numpy` arrays in `data/datasets/<dataset_name>/audio/`. Each file name should correspond to the `track_id` field in the annotations (e.g `track_id.npy`). A preprocessing example for wav files is provided below:

```python
import os
import librosa
import numpy as np

audio_dir = "data/datasets/audiocaption/audio"
for audio_file in os.listdir(audio_dir):
    audio_path = os.path.join(audio_dir, audio_file)
    audio, sr = librosa.load(audio_path, 16000)
    array_path = audio_path.replace("wav", "npy")
    np.save(open(array_path, 'wb'), audio)
```

If the audio file names do not correspond to the track IDs, you'll need an additional field in the annotation files containing the file paths. In this case, remember to edit the code in [`muscaps/datasets/`](../muscaps/datasets/) to point to the corresponding file paths when loading the data.
