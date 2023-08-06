'''
python3 test.py --data_root av-toy-preprocessed --checkpoint_path checkpoint/checkpoint_step000003000.pth
'''
from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import SyncNet_color as SyncNet
import audio

import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np

from glob import glob

import os, random, cv2, argparse
from hparams import hparams, get_image_list

parser = argparse.ArgumentParser(description='Code to test the expert lip-sync discriminator')

parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True)
parser.add_argument("--video_id", help="Test video id", required=True)
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint', default=None, type=str)


args = parser.parse_args()

use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16

class Dataset(object):
    def __init__(self, idx):
        sub_folder = os.path.join(args.data_root, f"test/{idx}")
        img_names = glob(f"{sub_folder}/*.jpg")
        img_names.sort(key = lambda x: int(x.split('/')[-1][:-4]))
        self.img_names = img_names
        print(f"Test data size: {len(img_names)}")

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def crop_audio_window(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]


    def __len__(self):
        return len(self.img_names) - syncnet_T + 1

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        y = torch.ones(1).float()
        chosen = img_name
        vidname = dirname(chosen)

        window_fnames = self.get_window(chosen)
        if window_fnames is None: return

        window = []
        all_read = True
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                all_read = False
                break
            try:
                img = cv2.resize(img, (hparams.img_size, hparams.img_size))
            except Exception as e:
                all_read = False
                break

            window.append(img)

        if not all_read: return

        try:
            wavpath = join(vidname, "audio1.wav")
            wav = audio.load_wav(wavpath, hparams.sample_rate)

            orig_mel = audio.melspectrogram(wav).T
        except Exception as e:
            print(f"Error: {e}")
            return

        mel = self.crop_audio_window(orig_mel.copy(), img_name)

        if (mel.shape[0] != syncnet_mel_step_size):
            return

        mel = self.crop_audio_window(orig_mel.copy(), img_name)

        # H x W x 3 * T
        x = np.concatenate(window, axis=2) / 255.
        x = x.transpose(2, 0, 1)
        x = x[:, x.shape[1]//2:]

        x = torch.FloatTensor(x)
        mel = torch.FloatTensor(mel.T).unsqueeze(0)
        return x, mel, y

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss

def eval_model(test_data_loader, device, model):
    losses = []
    for x, mel, y in tqdm(test_data_loader):

        model.eval()

        # Transform data to CUDA device
        x = x.to(device)

        mel = mel.to(device)

        a, v = model(mel, x)
        y = y.to(device)

        loss = cosine_loss(a, v, y)
        losses.append(loss.item())

    averaged_loss = sum(losses) / len(losses)

    return averaged_loss

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model, optimizer, reset_optimizer=False):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    return model

if __name__ == "__main__":
    checkpoint_path = args.checkpoint_path

    # Dataset and Dataloader setup
    test_dataset = Dataset('0004')

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.syncnet_batch_size,
        num_workers=8, drop_last=True)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    model = SyncNet().to(device)
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.syncnet_lr)

    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)
        averaged_loss = eval_model(test_data_loader, device, model)
        print(f"Loss: {averaged_loss}")
