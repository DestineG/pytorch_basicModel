# chapter7-3-Tacotron2/data.py

import os
import csv
import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader

def build_textData_from_csv(meta_path: str):
    items = []
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) < 2:
                print("Malformed row:", row)
                continue
            items.append((row[0], row[1]))
    return items

def build_speech_map(data_dir: str):
    speech_map = {}
    for filename in os.listdir(data_dir):
        if filename.endswith(".wav"):
            utt_id = filename.replace(".wav", "")
            speech_map[utt_id] = os.path.join(data_dir, filename)
    return speech_map

def build_data(meta_path: str, speechData_dir: str):
    text_data = build_textData_from_csv(meta_path)
    speech_map = build_speech_map(speechData_dir)

    data = []
    for utt_id, text in text_data:
        if utt_id not in speech_map:
            print(f"Missing wav for {utt_id}")
            continue
        data.append((text, speech_map[utt_id]))

    return data

class TextProcessor:
    sepcial_tokens = {
        "pad": {
            "id": 0,
            "token": "<pad>"
        },
        "eos": {
            "id": 1,
            "token": "<eos>"
        }
    }
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!'(),-.:;? "
    vocab_size = len(chars) + len(sepcial_tokens)
    def __init__(self):
        # 定义字符集
        self.chars = TextProcessor.chars

        # 构建 vocab
        self.itos = [TextProcessor.sepcial_tokens["pad"]["token"], TextProcessor.sepcial_tokens["eos"]["token"]] + list(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @staticmethod
    def ids_padding(ids: torch.Tensor, max_len: int) -> torch.Tensor:
        padded_ids = torch.full((max_len,), fill_value=TextProcessor.sepcial_tokens["pad"]["id"], dtype=torch.long)
        length = min(len(ids), max_len)
        padded_ids[:length] = ids[:length]
        return padded_ids

    def text2ids(self, text: str) -> torch.Tensor:
        ids = [self.stoi[c] for c in text if c in self.stoi]
        ids.append(self.stoi[TextProcessor.sepcial_tokens["eos"]["token"]])
        return torch.tensor(ids, dtype=torch.long)

    def ids2text(self, ids: torch.Tensor) -> str:
        return "".join([self.itos[i] for i in ids if i != TextProcessor.sepcial_tokens["pad"]["id"] and i != TextProcessor.sepcial_tokens["eos"]["id"]])

class WavProcessor:
    def __init__(self,
                 sr=22050,              # 音频采样率
                 n_fft=1024,            # 傅里叶变换窗口长度
                 hop_length=256,        # 相邻傅里叶变换窗口步长
                 n_mels=80              # 梅尔滤波器数量(每帧特征维度)
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def wav2mel(self, wav_path: str) -> torch.FloatTensor:
        # 1. 读取音频
        y, sr = librosa.load(wav_path, sr=self.sr)
        # 2. 归一化
        y = y / max(abs(y) + 1e-9)  # 防止除零
        # 3. 计算 Mel-spectrogram
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        # 4. 转为 Tensor
        return torch.tensor(mel_db, dtype=torch.float32)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        meta_path = os.path.join(data_dir, "metadata.csv")
        speechData_dir = os.path.join(data_dir, "wavs")
        self.data = build_data(meta_path, speechData_dir)
        self.text_processor = TextProcessor()
        self.wav_processor = WavProcessor()
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, wav_path = self.data[idx]
        # text -> ids
        text = self.text_processor.text2ids(text)
        # wav_path -> mel
        wav = self.wav_processor.wav2mel(wav_path)
        return text, wav
    
    def collate_fn(self, batch):
        texts, wavs = zip(*batch)
        max_text_len = max([len(t) for t in texts])
        max_wav_len = max([w.shape[1] for w in wavs])

        padded_texts = torch.stack([
            self.text_processor.ids_padding(t, max_text_len)
            for t in texts
        ], dim=0)
        texts_len = torch.tensor([len(t) for t in texts], dtype=torch.long)

        padded_wavs = torch.stack([
            torch.nn.functional.pad(w, (0, max_wav_len - w.shape[1]), "constant", 0.0)
            for w in wavs
        ], dim=0)
        wavs_len = torch.tensor([w.shape[1] for w in wavs], dtype=torch.long)

        return padded_texts, texts_len, padded_wavs, wavs_len

def build_dataloader(batch_size: int, shuffle: bool = True):
    data_dir = "/dataroot/liujiang/data/datasets/LJSpeech1.1"
    dataset = Dataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=dataset.collate_fn)
    return dataloader

if __name__ == "__main__":
    dataloader = build_dataloader(batch_size=4, shuffle=True)
    for batch in dataloader:
        texts, texts_len, wavs, wavs_len = batch
        print("Texts shape:", texts.shape)  # (B, T_text)
        print("Texts lengths:", texts_len)  # (B,)
        print("Wavs shape:", wavs.shape)    # (B, n_mels, T_wav)
        print("Wavs lengths:", wavs_len)    # (B,)
        break

# Texts shape: torch.Size([4, 106])
# Wavs shape: torch.Size([4, 80, 758])
# Texts lengths: tensor([ 48,  80,  46, 106])
# Wavs lengths: tensor([758, 424, 311, 619])