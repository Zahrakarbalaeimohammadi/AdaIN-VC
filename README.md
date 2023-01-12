# AdaIN-VC

This is an unofficial implementation of the paper [One-shot Voice Conversion by Separating Speaker and Content Representations with Instance Normalization](https://arxiv.org/abs/1904.05742) modified from the official one.

## Dependencies

- Python >= 3.6
- torch >= 1.7.0
- torchaudio >= 0.7.0
- numpy >= 1.16.0
- librosa >= 0.6.3

## Differences from the official implementation

The main difference from the official implementation is the use of a neural vocoder, which greatly improves the audio quality.
I adopted universal vocoder, whose code was from [yistLin/universal-vocoder](https://github.com/yistLin/universal-vocoder) and checkpoint will be available soon.
Besides, this implementation supports torch.jit, so the full model can be loaded with simply one line:

```python
model = torch.jit.load(model_path)
```

Pre-trained models are available [here](https://drive.google.com/drive/folders/1MacKgXGA4Ad0O_c6W5MlkZMG0B8IzaM-?usp=sharing).

## Preprocess

The code `preprocess.py` extracts features from raw audios.

```bash
python preprocess.py <data_dir> <save_dir> [--segment seg_len]
```

- **data_dir**: The directory of speakers.
- **save_dir**: The directory to save the processed files.
- **seg_len**: The length of segments for training.

## Training

```bash
python train.py <config_file> <data_dir> <save_dir> [--n_steps steps] [--save_steps save] [--log_steps log] [--n_spks spks] [--n_uttrs uttrs]
```

- **config_file**: The config file for AdaIN-VC.
- **data_dir**: The directory of processed files given by `preprocess.py`.
- **save_dir**: The directory to save the model.
- **steps**: The number of steps for training.
- **save**: To save the model every <em>save</em> steps.
- **log**: To record training information every <em>log</em> steps.
- **spks**: The number of speakers in the batch.
- **uttrs**: The number of utterances for each speaker in the batch.

## Inference

You can use `inference.py` to perform one-shot voice conversion.
The pre-trained model will be available soon.

```bash
python inference.py <model_path> <vocoder_path> <source> <target> <output>
```

- **model_path**: The path of the model file.
- **vocoder_path**: The path of the vocoder file.
- **source**: The utterance providing linguistic content.
- **target**: The utterance providing target speaker timbre.
- **output**: The converted utterance.

## Reference

Please cite the paper if you find AdaIN-VC useful.

```bib
@article{chou2019one,
  title={One-shot voice conversion by separating speaker and content representations with instance normalization},
  author={Chou, Ju-chieh and Yeh, Cheng-chieh and Lee, Hung-yi},
  journal={arXiv preprint arXiv:1904.05742},
  year={2019}
}
```
## Sections added by Zahra Karbalaei Mohammadi
## 1. A summary of the purpose as well as the function of the code

In this research, an algorithm for changing the voice of people has been presented, which can easily take the voice of a person with a specific content and have the voice of another person with a different content next to it, and at the end, the first content with the voice of the second person as the output. to give
The function of the code is completely algorithmic and does not have any input or output as sound. Therefore, in the future, they can use this algorithm in voice conversion tasks.


## 2. The rate of innovation in code improvement

Due to the fact that this project did not have any input and output, I found a similar work in the link https://colab.research.google.com/github/yiftachbeer/AdaIN-VC/blob/master/notebooks/demo.ipynb#scrollTo=RmNTzr2Fds5l which can show the performance of this project well.
Also, according to the studies, I found that if AdaIN-VC is used with AGAIN-VC, it can greatly reduce the dimensions and prevent the speaker information from leaking into the content embeddings.


## 3. Things that have been changed and improved in the source code

There were no specific bugs in the original source code, and most of the problems were related to pycodestyle, which I tried to fix as much as possible.
New source code to show the performance of the AdaIN-VC algorithm:

# AdaIN-VC demo
This is a demonstration of AdaIN-VC that should work out of the box and be fairly quick to setup.

## Code Setup
!git clone https://github.com/yiftachbeer/AdaIN-VC
%cd AdaIN-VC
%%capture

!python -m pip install -r requirements.txt
## Data Setup
#We download a custom, smaller version of VCTK (all utterances of 5 speakers out of 110).
%%capture

!wget https://www.cs.huji.ac.il/~yiftach/VCTKmini.zip
!unzip VCTKmini.zip && rm VCTKmini.zip

%run adain-vc.py preprocess VCTKmini/wav48_silence_trimmed VCTKmini_mel

## Training
#The `n_steps` parameter can be adjusted depending on how long you want to wait. 

%run adain-vc.py train config.yaml VCTKmini_mel saved_models --n_steps 1000 --save_steps 100

## Inference
from IPython.display import Audio
#We use the first sample for content, and the second for speaker:
Audio('VCTKmini/p226/p226_002_mic2.flac')
Audio('VCTKmini/p225/p225_003_mic2.flac')

#We demonstrate the quality of the pretrained model along with the one we just trained:
%run adain-vc.py inference VCTKmini/p226/p226_002_mic2.flac VCTKmini/p225/p225_003_mic2.flac cvrt-trained.wav --model_path saved_models/model-1000.ckpt
%run adain-vc.py inference VCTKmini/p226/p226_002_mic2.flac VCTKmini/p225/p225_003_mic2.flac cvrt-pretrained.wav
Audio('cvrt-trained.wav')
Audio('cvrt-pretrained.wav')

Reference to the project: https://colab.research.google.com/github/yiftachbeer/AdaIN-VC/blob/master/notebooks/demo.ipynb#scrollTo=KVcBHvvfV9s-


## 4. The result of changing and improving the code in evaluating the output audio file

After solving these problems, no other problems were observed in Visual Studio Code, but because the work is coded as an algorithm, we had no sound input or output.
I was only able to check the codes and fix the errors and problems according to the Visual Studio Code guidelines to improve the coding.


## 5. Reference to the main project link

https://github.com/cyhuang-tw/AdaIN-VC


## 6. Student introduction

Zahra karbalaei mohammadi is a master's student from South Tehran University

student number: 40014140111030

Digital signal processing course

Supervisor: Dr. Mahdi Eslami


## 7. The article file has been updated

Link to download the comparison table of advantages and disadvantages of the method used in 10 articles with similar topics: https://drive.google.com/file/d/1senPl-zaLvEdadwrADIY5Ibc84KInL_0/view?usp=share_link

Download link of the introduction of the new article: https://drive.google.com/file/d/1DQQAOIRcSbO8AzGZWcIVnzyrnj3OjqAG/view?usp=share_link


## 8. Explanation videos about project code and articles

Video file link for a general explanation about the article: https://drive.google.com/file/d/1Dj-tNs13g7Z3m3rBQ18mCPBXiyZSz6KJ/view?usp=sharing

Video link for a detailed explanation of the article: https://drive.google.com/file/d/1UAlZrxqV7mTjHeamGqRVoJ2YErcp-zPM/view?usp=sharing

Video file general explanation about the main parts of the source and code database and the environment and software required to run the code: https://drive.google.com/file/d/1wytASSQn8NkKPPeb_t8BlzIFMRxnuWVC/view?usp=sharing

Video file link explaining about the code and matching it with the article: https://drive.google.com/file/d/18sz51-JWXAwIVSdhsA0BptDeGtr9HQTd/view?usp=sharing

Link to the video file of the source code execution and explanation about the input and output of the final project: https://drive.google.com/file/d/1U7PZy5zF4mX2T4OMar-OgltCig1yvK4V/view?usp=drivesdk

The link of the input and output file in another similar project that uses the algorithm of my final project: https://drive.google.com/drive/folders/1fY-dxzlGMZGa0sGDEVHKmRQQNKZ9tLFq


All the videos related to my project to promote science have been uploaded to Aparat

Aparat link: https://www.aparat.com/Zahrakarbalaeimohammadi


## 9. Completed proposal file for the project

Download link: 

https://drive.google.com/file/d/10ofAt50bEqUU1LrLRz96W2oYHmpbMS94/view?usp=share_link

## All the tasks done for the progress of the project

Download link: 

https://drive.google.com/drive/folders/11PUIKewGp8iGzwEpmLxETaU9BjT11w9k?usp=share_link

## The link to the final project presentation

Download link:


