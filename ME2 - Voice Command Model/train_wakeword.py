import os
import time
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import soundfile as sf
import pandas as pd
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = "/home/misael.andre.maningo/MEng AI/AI 231/ME2 - Voice Command Model"
MANIFEST_PATH = os.path.join(BASE_DIR, "data/wakeword/wakeword_manifest.csv")
TARGET_SR = 16000
TARGET_SAMPLES = 16000 # 1.0s

class RobustWakeNet(nn.Module):
    """
    Robust Depthwise Separable CNN for Wake Word Spotting.
    ~38k parameters, <2.0ms on CPU.
    """
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        def dw_block(c_in, c_out, stride=1):
            return nn.Sequential(
                nn.Conv2d(c_in, c_in, kernel_size=3, stride=stride, padding=1, groups=c_in, bias=False),
                nn.BatchNorm2d(c_in),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_in, c_out, kernel_size=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            )
            
        self.blocks = nn.Sequential(
            dw_block(32, 48, stride=1),
            dw_block(48, 64, stride=(2, 2)),
            dw_block(64, 64, stride=1)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits


class RobustWakeDataset(Dataset):
    def __init__(self, df, base_dir, is_train=False):
        self.df = df.reset_index(drop=True)
        self.base_dir = base_dir
        self.is_train = is_train
        
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=TARGET_SR,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=40
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB()
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=15)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=6)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.base_dir, row["path"])
        data, sr = sf.read(file_path, dtype="float32")
        waveform = torch.from_numpy(data)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t().mean(dim=0, keepdim=True)
            
        if sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
            waveform = resampler(waveform)
            
        # Audio domain augmentations for training
        if self.is_train:
            # 1. Random Gain (0.3x to 1.3x) to handle distance and mic sensitivity
            gain = random.uniform(0.3, 1.3)
            waveform = waveform * gain
            
            # 2. Random Time Shift (+/- 2400 samples = +/- 150ms)
            shift = random.randint(-2400, 2400)
            if shift > 0:
                waveform = torch.nn.functional.pad(waveform, (shift, 0))[:, :-shift]
            elif shift < 0:
                waveform = torch.nn.functional.pad(waveform, (0, -shift))[:, -shift:]

        length = waveform.shape[-1]
        if length < TARGET_SAMPLES:
            pad = TARGET_SAMPLES - length
            waveform = torch.nn.functional.pad(waveform, (pad // 2, pad - pad // 2))
        elif length > TARGET_SAMPLES:
            start = (length - TARGET_SAMPLES) // 2
            waveform = waveform[:, start : start + TARGET_SAMPLES]
            
        mel = self.melspec(waveform)
        log_mel = self.amp_to_db(mel)
        
        if self.is_train:
            log_mel = self.time_mask(self.freq_mask(log_mel))
            
        label = int(row["label_idx"])
        return log_mel, label


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.numpy())
            all_probs.extend(probs)
            
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    neg_mask = (all_targets == 0)
    far = (all_preds[neg_mask] == 1).mean() if neg_mask.sum() > 0 else 0.0
    pos_mask = (all_targets == 1)
    frr = (all_preds[pos_mask] == 0).mean() if pos_mask.sum() > 0 else 0.0
    
    return acc, prec, rec, f1, far, frr, all_targets, all_preds, all_probs


def main():
    print("=" * 60)
    print("RETRAINING ROBUST WAKE WORD MODEL ('Hey Raspberry')")
    print("Augmentations: Random Time Shift (+/-150ms), Gain (0.3-1.3x), SpecAugment")
    print("=" * 60)
    
    df = pd.read_csv(MANIFEST_PATH)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    
    train_ds = RobustWakeDataset(train_df, BASE_DIR, is_train=True)
    val_ds = RobustWakeDataset(val_df, BASE_DIR, is_train=False)
    test_ds = RobustWakeDataset(test_df, BASE_DIR, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RobustWakeNet().to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {params:,} ({params * 4 / 1024:.1f} KB in FP32)")
    
    # Positive weight of 2.8x balances the 3:1 negative ratio
    pos_weight = torch.tensor([1.0, 2.8]).to(device)
    criterion = nn.CrossEntropyLoss(weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-5)
    
    best_val_f1 = 0.0
    ckpt_path = os.path.join(BASE_DIR, "checkpoints/best_wakeword.pt")
    
    for epoch in range(1, 26):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            
        scheduler.step()
        train_loss /= len(train_loader.dataset)
        val_acc, val_prec, val_rec, val_f1, val_far, val_frr, _, _, _ = evaluate(model, val_loader, device)
        
        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), ckpt_path)
            
        print(f"Epoch [{epoch:02d}/25] | Train Loss: {train_loss:.4f} | "
              f"Val Acc: {val_acc*100:.1f}% Rec: {val_rec*100:.1f}% Prec: {val_prec*100:.1f}% F1: {val_f1:.4f} "
              f"{'[*BEST*]' if is_best else ''}")
              
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON UNSEEN WAKE WORD TEST SET")
    print("=" * 60)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    test_acc, test_prec, test_rec, test_f1, test_far, test_frr, y_true, y_pred, test_probs = evaluate(model, test_loader, device)
    
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Precision:     {test_prec*100:.2f}%")
    print(f"Recall:        {test_rec*100:.2f}%")
    print(f"Macro F1:      {test_f1:.4f}")
    print(f"False Acceptance Rate (FAR): {test_far*100:.2f}%")
    print(f"False Rejection Rate (FRR):  {test_frr*100:.2f}%")
    print(f"Average Positive Wake Confidence: {test_probs[y_true==1].mean()*100:.1f}%")
    print(f"Average Negative Score:           {test_probs[y_true==0].mean()*100:.1f}%")
    
    # Export to ONNX
    print("\n--- Exporting to ONNX & Quantizing ---")
    model.eval().to("cpu")
    dummy_input = torch.randn(1, 1, 40, 101, dtype=torch.float32)
    fp32_onnx = os.path.join(BASE_DIR, "exports/wakeword_fp32.onnx")
    int8_onnx = os.path.join(BASE_DIR, "exports/wakeword_int8.onnx")
    
    torch.onnx.export(
        model,
        dummy_input,
        fp32_onnx,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        dynamo=False
    )
    print(f"Exported FP32 ONNX: {fp32_onnx} ({os.path.getsize(fp32_onnx)/1024:.1f} KB)")
    
    quantize_dynamic(
        model_input=fp32_onnx,
        model_output=int8_onnx,
        weight_type=QuantType.QInt8
    )
    print(f"Exported INT8 ONNX: {int8_onnx} ({os.path.getsize(int8_onnx)/1024:.1f} KB)")
    
    # Benchmark CPU latency
    session = ort.InferenceSession(int8_onnx, providers=["CPUExecutionProvider"])
    inp_name = session.get_inputs()[0].name
    dummy_np = dummy_input.numpy()
    for _ in range(50):
        _ = session.run(None, {inp_name: dummy_np})
    t0 = time.perf_counter()
    for _ in range(500):
        _ = session.run(None, {inp_name: dummy_np})
    latency = (time.perf_counter() - t0) / 500 * 1000.0
    print(f"INT8 CPU Latency: {latency:.2f} ms per 1.0s window.")

if __name__ == "__main__":
    main()
