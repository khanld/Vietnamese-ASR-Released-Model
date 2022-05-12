# Vietnamese Speech Recognition using Wav2vec 2.0
### Table of contents
1. [Model Description](#description)
2. [Implementation](#implementation)
3. [Benchmark Result](#benchmark)
4. [Example Usage](#example)
5. [Evaluation](#evaluation)
6. [Citation](#citation)
7. [Contact](#contact)
<a name = "description" ></a>
### Model Description
Fine-tuned the Wav2vec2-based model on about 160 hours of Vietnamese speech dataset from different resources, including [VIOS](https://huggingface.co/datasets/vivos), [COMMON VOICE](https://huggingface.co/datasets/mozilla-foundation/common_voice_8_0), [FOSD](https://data.mendeley.com/datasets/k9sxg2twv4/4) and [VLSP 100h](https://drive.google.com/file/d/1vUSxdORDxk-ePUt-bUVDahpoXiqKchMx/view). We have not yet incorporated the Language Model into our ASR system but still gained a promising result.
<a name = "implementation" ></a>
### Implementation
We also provide code for Pre-training and Fine-tuning the Wav2vec2 model. If you wish to train on your dataset, check it out here:
- [Pre-train code](https://github.com/khanld/ASR-Wav2vec-Pretrain) (not available for now but will release soon)
- [Fine-tune code](https://github.com/khanld/ASR-Wa2vec-Finetune)

<a name = "benchmark" ></a>
### Benchmark WER Result
| | [VIVOS](https://huggingface.co/datasets/vivos) | [COMMON VOICE 8.0](https://huggingface.co/datasets/mozilla-foundation/common_voice_8_0) | 
|---|---|---|
|without LM| 15.05 | 10.78 |
|with LM| in progress | in progress |

<a name = "example" ></a>
### Example Usage [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1blz1KclnIfbOp8o2fW3WJgObOQ9SMGBo?usp=sharing)
```python
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Wav2Vec2Processor.from_pretrained("khanhld/wav2vec2-base-vietnamese-160h")
model = Wav2Vec2ForCTC.from_pretrained("khanhld/wav2vec2-base-vietnamese-160h")
model.to(device)
def transcribe(wav):
  input_values = processor(wav, sampling_rate=16000, return_tensors="pt").input_values
  logits = model(input_values.to(device)).logits
  pred_ids = torch.argmax(logits, dim=-1)
  pred_transcript = processor.batch_decode(pred_ids)[0]
  return pred_transcript
wav, _ = librosa.load('path/to/your/audio/file', sr = 16000)
print(f"transcript: {transcribe(wav)}")
```

<a name = "evaluation"></a>
### Evaluation [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XQCq4YGLnl23tcKmYeSwaksro4IgC_Yi?usp=sharing)

```python
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import torch
import re
from datasets import load_dataset, load_metric, Audio
wer = load_metric("wer")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load processor and model
processor = Wav2Vec2Processor.from_pretrained("khanhld/wav2vec2-base-vietnamese-160h")
model = Wav2Vec2ForCTC.from_pretrained("khanhld/wav2vec2-base-vietnamese-160h")
model.to(device)
model.eval()
# Load dataset
test_dataset = load_dataset("mozilla-foundation/common_voice_8_0", "vi", split="test", use_auth_token="your_huggingface_auth_token")
test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))
chars_to_ignore = r'[,?.!\-;:"“%\'�]' # ignore special characters
# preprocess data
def preprocess(batch):
  audio = batch["audio"]
  batch["input_values"] = audio["array"]
  batch["transcript"] = re.sub(chars_to_ignore, '', batch["sentence"]).lower()
  return batch
# run inference
def inference(batch):
  input_values = processor(batch["input_values"], 
                            sampling_rate=16000, 
                            return_tensors="pt").input_values
  logits = model(input_values.to(device)).logits
  pred_ids = torch.argmax(logits, dim=-1)
  batch["pred_transcript"] = processor.batch_decode(pred_ids) 
  return batch
  
test_dataset = test_dataset.map(preprocess)
result = test_dataset.map(inference, batched=True, batch_size=1)
print("WER: {:2f}".format(100 * wer.compute(predictions=result["pred_transcript"], references=result["transcript"])))
```
**Test Result**: 10.78%

<a name = "citation" ></a>
### Citation 
[![DOI](https://zenodo.org/badge/485623832.svg)](https://github.com/khanld/ASR-Wa2vec-Finetune)
```text
@misc{Khanhld_Vietnamese_Wav2vec_Asr_2022,
  author = {Duy Khanh Le},
  doi = {10.5281/zenodo.6540979},
  month = {May},
  title = {Finetune Wav2vec 2.0 For Vietnamese Speech Recognition},
  url = {https://github.com/khanld/ASR-Wa2vec-Finetune},
  year = {2022}
}
```

<a name = "contact"></a>
### Contact
- khanhld218@uef.edu.vn
- [![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/)
- [![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/khanhld257/)

