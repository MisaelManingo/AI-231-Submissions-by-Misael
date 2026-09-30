import os
import random
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
import pandas as pd
import numpy as np

class SpecAugment(nn.Module):
    """
    SpecAugment: Time and Frequency Masking for Mel-Spectrograms.
    """
    def __init__(self, freq_mask_param=8, time_mask_param=20):
        super().__init__()
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param)

    def forward(self, spec):
        return self.time_mask(self.freq_mask(spec))


class VoiceCommandDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        manifest_df,
        base_dir,
        target_sr=16000,
        target_samples=32000,
        is_train=False,
        use_specaugment=True
    ):
        self.df = manifest_df.reset_index(drop=True)
        self.base_dir = base_dir
        self.target_sr = target_sr
        self.target_samples = target_samples
        self.is_train = is_train
        self.use_specaugment = is_train and use_specaugment
        
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=40
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_augment = SpecAugment(freq_mask_param=8, time_mask_param=20)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.base_dir, row["path"])
        
        # Robust loading via soundfile
        data, sr = sf.read(file_path, dtype="float32")
        waveform = torch.from_numpy(data)
        
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t().mean(dim=0, keepdim=True)
            
        # Resample if not 16kHz
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
            
        length = waveform.shape[-1]
        
        # Audio domain augmentations for training
        if self.is_train:
            # Random time shift (+/- 0.1s = 1600 samples)
            shift = random.randint(-1600, 1600)
            if shift > 0:
                waveform = torch.nn.functional.pad(waveform, (shift, 0))[:, :-shift]
            elif shift < 0:
                waveform = torch.nn.functional.pad(waveform, (0, -shift))[:, -shift:]

        # Fit to target length (32,000 samples)
        length = waveform.shape[-1]
        if length < self.target_samples:
            pad_total = self.target_samples - length
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            waveform = torch.nn.functional.pad(waveform, (pad_left, pad_right))
        elif length > self.target_samples:
            if self.is_train:
                max_start = length - self.target_samples
                start = random.randint(0, max_start)
            else:
                start = (length - self.target_samples) // 2
            waveform = waveform[:, start : start + self.target_samples]
            
        # Log-Mel Spectrogram extraction: (1, 40, 201)
        mel = self.melspec(waveform)
        log_mel = self.amp_to_db(mel)
        
        # Apply SpecAugment if training
        if self.use_specaugment:
            log_mel = self.spec_augment(log_mel)
            
        label = int(row["label_idx"])
        return log_mel, label
