import os
import sys
import json
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

from dataset import VoiceCommandDataset
from models.dscnn import get_dscnn
from models.bcresnet import get_bcresnet

BASE_DIR = "/home/misael.andre.maningo/MEng AI/AI 231/ME2 - Voice Command Model"
OPTIONB_DIR = os.path.join(BASE_DIR, "upstream_repo/MEX2/OptionB")
MANIFEST_PATH = os.path.join(BASE_DIR, "data/unified_manifest.csv")
LABELS_PATH = os.path.join(BASE_DIR, "data/labels_20.json")

def parse_args():
    parser = argparse.ArgumentParser(description="Train Tiny Voice Command Model (VCM)")
    parser.add_argument("--model", type=str, default="dscnn", choices=["dscnn", "bcresnet"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    return parser.parse_args()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, macro_f1, np.array(all_targets), np.array(all_preds)


def main():
    args = parse_args()
    print("=" * 60)
    print(f"Training Tiny Voice Command Model: {args.model.upper()}")
    print("=" * 60)
    
    # Load label map
    with open(LABELS_PATH, "r") as f:
        label_info = json.load(f)
    idx2label = {int(k): v for k, v in label_info["idx2label"].items()}
    num_classes = len(idx2label)
    class_names = [idx2label[i] for i in range(num_classes)]
    
    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    # Load Manifest
    df = pd.read_csv(MANIFEST_PATH)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    print(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Datasets and Loaders
    train_ds = VoiceCommandDataset(train_df, OPTIONB_DIR, is_train=True, use_specaugment=True)
    val_ds = VoiceCommandDataset(val_df, OPTIONB_DIR, is_train=False)
    test_ds = VoiceCommandDataset(test_df, OPTIONB_DIR, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # Model instantiation
    if args.model == "dscnn":
        model = get_dscnn(num_classes=num_classes)
    elif args.model == "bcresnet":
        model = get_bcresnet(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {args.model}")
        
    param_count = count_parameters(model)
    print(f"Model parameters: {param_count:,} ({param_count * 4 / (1024 * 1024):.2f} MB in FP32)")
    model = model.to(device)
    
    # Loss, Optimizer, Scheduler
    # Label smoothing = 0.05 prevents overconfident logits on noisy samples
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    
    best_val_f1 = 0.0
    checkpoint_path = os.path.join(BASE_DIR, "checkpoints", f"best_{args.model}.pt")
    
    # Training Loop
    print("\nStarting training loop...")
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        epoch_start = time.time()
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(targets.cpu().numpy())
            
        scheduler.step()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, average="macro", zero_division=0)
        
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start
        
        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_f1": val_f1,
                "model_name": args.model,
                "num_classes": num_classes
            }, checkpoint_path)
            
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_time:.1f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% F1: {val_f1:.4f} "
              f"{'[*BEST SAVED*]' if is_best else ''}")
              
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes.")
    print(f"Best Validation Macro F1: {best_val_f1:.4f}")
    
    # Evaluate best checkpoint on Test Split
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON UNSEEN SPEAKER-DISJOINT TEST SET (10 Speakers)")
    print("=" * 60)
    checkpoint = torch.save if False else torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Top-1 Accuracy: {test_acc*100:.2f}%")
    print(f"Test Macro F1: {test_f1:.4f}")
    
    # Class-wise report
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\nDetailed Per-Class Performance:")
    print(report)
    
    # Background rejection rate
    bg_mask = (y_true == 0)
    bg_rejection_acc = accuracy_score(y_true[bg_mask], y_pred[bg_mask])
    print(f"Negative / Background Rejection Rate: {bg_rejection_acc*100:.2f}% ({bg_mask.sum()} samples)")
    
    # Save test results summary to json
    results_path = os.path.join(BASE_DIR, "exports", f"test_metrics_{args.model}.json")
    with open(results_path, "w") as f:
        json.dump({
            "model": args.model,
            "parameters": param_count,
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "test_macro_f1": float(test_f1),
            "bg_rejection_rate": float(bg_rejection_acc)
        }, f, indent=2)
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()
