# DataLoader
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchaudio
import torch.nn as nn
from transformers import BertTokenizer

import cv2
import os
import numpy as np
import pandas as pd

class MELDDataset(Dataset):
    def __init__(self, csv, path, transform=None, max_video_len=30, max_audio_len=16000, max_text_len=128):
        self.df = pd.read_csv(csv)
        self.label = {self.df['Emotion'].unique()[i] : i for i in range(len(self.df['Emotion'].unique()))}
        self.path = path
        self.max_video_len = max_video_len
        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.video_transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def load_video(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.video_transform(frame)
            frames.append(frame)
            if len(frames) >= self.max_video_len:
                break
        cap.release()
        frames = frames[:self.max_video_len]
        if len(frames) < self.max_video_len:
            frames.extend([torch.zeros_like(frames[0])] * (self.max_video_len - len(frames)))
        return torch.stack(frames)

    def load_audio(self, audio_path):
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if waveform.size(1) > self.max_audio_len:
                waveform = waveform[:, :self.max_audio_len]
            else:
                padding = self.max_audio_len - waveform.size(1)
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            return waveform
        except:
            return torch.zeros(1, self.max_audio_len)


    def tokenize_text(self, text):
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_text_len, return_tensors='pt')
        return encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze()
    
    def __getitem__(self, idx):
        filename = 'dia' + str(self.df.iloc[idx]['Dialogue_ID']) + '_utt' + str(self.df.iloc[idx]['Utterance_ID']) + '.mp4'
        text = self.df.iloc[idx]['Utterance'].replace('\x92', "'")

        video = self.load_video(self.path + filename)
        audio = self.load_audio(self.path + filename)
        text, attention_mask = self.tokenize_text(text)
        label = self.label[self.df.iloc[idx]['Emotion']]
        return video, audio, text, attention_mask, label

def collate_fn(batch):
    videos, audios, texts, attention_masks, labels = zip(*batch)
    videos = torch.stack(videos)
    audios = torch.stack(audios)
    texts = torch.stack(texts)
    attention_masks = torch.stack(attention_masks)
    labels = torch.tensor(labels)
    return videos, audios, texts, attention_masks, labels

def MELD(datatype, transform=None, batch_size=4, collate=collate_fn):
    """DataLoader. \\
    Expected File structure is: \\
    ├── train\\
    ├── valid\\
    ├── test  \\
    ├── train.csv\\
    ├── valid.csv\\
    └── test.csv\\
    Change if you want. \\
    If transform is None, it just resizes data and returns Tensor.\\
    Video (Batch, Frame, Channel, Height, Width) \\
    Audio (Batch, Channel, Sample) \\
    Text  (Batch, tokenized Length)\\
    Label (Batch)
    """
    # Data to load
    if datatype == 'train':
        csv_file = './MELD_Data/train.csv'
        data_folder = './MELD_Data/train/'
    elif datatype == 'valid':
        csv_file = './MELD_Data/valid.csv'
        data_folder = './MELD_Data/valid/'
    elif datatype == 'test':
        csv_file = './MELD_Data/test.csv'
        data_folder = './MELD_Data/test/'
    # transform
    if transform is None:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    # Load data
    dataset = MELDDataset(csv_file, data_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    return dataloader
    
if __name__ == '__main__':
    dataloader = MELD('train', batch_size=4)

    for batch in dataloader:
        videos, audios, texts, attention_masks, labels = batch
        print(f"Video (B, F, C, H, W) : {videos.shape}")
        print(f"Audio (B, C, S) : {audios.shape}")
        print(f"Text  (B, L) : {texts.shape}")
        print(f"Label (B) : {labels.shape}")
        break