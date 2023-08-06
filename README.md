# Lip-Sync

## Theory
For sync between audio and videos of talking faces, I use two separate encoders for face and audio encoding, both comprising a stack of 2D-convolutions with residual skip connections. The input of the face encoder is a window of consecutive face frames, while the input of the audio encoder is the melspectrogram of a speech segment. The outputs of both encoders are 512p dimensional arrays. For the loss function, I use the cosine-similarity with binary cross-entropy loss. To be specific, I compute a dot product between the RELU-activated video and speech embedding to yield a single valye between [0, 1] for each sample that indicates the probability that the input audio-video pair is in sync.
This idea is referenced from [A Lip Sync Expert Is All You Need for Speech to Lip Generation In The Wild](https://arxiv.org/abs/2008.10010) and [Out of time: automated lip sync
in the wild](https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16a/chung16a.pdf)


The negative pairs are created by shifting the audio sampling window by several seconds compared to the video sampling window. In this way, out-of-sync pairs can be generated. As mentioned above, I will use the cosine-similarity with bianary cross entropy loss between the two embeddings to measure syncness.


## Implementation
I use [this](https://github.com/1adrianb/face-alignment) library for face extraction and alignment. For audio preprocessing, I use [librosa](https://librosa.org/doc/latest/index.html) for [front voice extraction](https://librosa.org/librosa_gallery/auto_examples/plot_vocal_separation.html) and melspectrogram extraction. Then, I split the dataset into 4:1 training set and validation set. I trained on 3070 for about an hour with a batch size of 4. One thing to notice is that the training and testing videos have different audio sample rates. The code is largedly based on [this](https://github.com/Rudrabha/Wav2Lip#training-on-datasets-other-than-lrs2) implementation.

Install Dependency
```
conda create --name lip_sync_env --file requirements.txt
conda activate lip_sync_env
```
Data preprocessing
```
python3 preprocess.py --ngpu 1 --batch_size 4 --data_root av-toy --preprocessed_root av-toy-preprocessed
python3 audio_segmentation.py
```
Model training
```
python3 train.py --data_root av-toy-preprocessed --checkpoint_dir checkpoint --error_log_path error.txt
```
Model testing
```
python3 test.py --data_root av-toy-preprocessed --checkpoint_path checkpoint/checkpoint_step000003000.pth
```


## Results
The training, evaluation and testing results can be found under the results folder. I used epoch size of 4 while training. In train_error.txt, each line contains the step number, the running average of the total loss so far, and the loss at each step. In validation_error.txt, each line contains the step number and the validation loss at each step. I took the model with the lowest validation error for testing. The pretrained model can be found [here](https://drive.google.com/file/d/1XUmQhtqeaekKuTgWowEY2tX-j4rjUKVa/view?usp=share_link). In the original paper, the loss is expected to be $\approx 0.25$, which is trained on a much larger [dataset](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html).