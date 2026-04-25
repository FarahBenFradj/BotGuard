import torch, torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np, os, json, pickle
from sklearn.metrics import f1_score, roc_auc_score

from src.preprocess import load_data
from src.model import BotDetector

DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH     = 32
EPOCHS    = 30
LR        = 1e-3
PATIENCE  = 5


def _run_epoch(model, loader, optimizer, criterion, train=True):
    model.train() if train else model.eval()
    loss_sum = correct = total = 0
    all_probs, all_labels = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for texts, meta, labels in loader:
            texts, meta, labels = texts.to(DEVICE), meta.to(DEVICE), labels.to(DEVICE)
            if train: optimizer.zero_grad()
            probs, _ = model(texts, meta)
            loss = criterion(probs, labels)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            loss_sum += loss.item()
            correct  += ((probs > 0.5).float() == labels).sum().item()
            total    += labels.size(0)
            all_probs.extend(probs.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return (loss_sum/len(loader), correct/total,
            np.array(all_probs), np.array(all_labels))


def train(data_dir='data/youtube', save_dir='models'):
    os.makedirs(save_dir, exist_ok=True)
    print(f"Device: {DEVICE}\n")

    tr_ds, val_ds, te_ds, tok, scaler, meta_cols = load_data(data_dir)
    tr_loader  = DataLoader(tr_ds,  batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH)
    te_loader  = DataLoader(te_ds,  batch_size=BATCH)

    model = BotDetector(
        vocab_size=len(tok), embed_dim=128, hidden_dim=64,
        meta_dim=len(meta_cols)
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.BCELoss()

    history   = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
    best_acc  = 0
    no_improve = 0

    for epoch in range(1, EPOCHS+1):
        tr_loss, tr_acc, _, _ = _run_epoch(model, tr_loader, optimizer, criterion, True)
        va_loss, va_acc, _, _ = _run_epoch(model, val_loader, optimizer, criterion, False)
        scheduler.step(va_loss)

        for k, v in zip(['train_loss','val_loss','train_acc','val_acc'],
                        [tr_loss, va_loss, tr_acc, va_acc]):
            history[k].append(round(v, 4))

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"tr_loss {tr_loss:.4f} tr_acc {tr_acc:.4f} | "
              f"val_loss {va_loss:.4f} val_acc {va_acc:.4f}")

        if va_acc > best_acc:
            best_acc   = va_acc
            no_improve = 0
            torch.save(model.state_dict(), f"{save_dir}/best_model.pt")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n⏹  Early stopping at epoch {epoch}")
                break

    # Test
    model.load_state_dict(torch.load(f"{save_dir}/best_model.pt", map_location=DEVICE))
    _, te_acc, te_probs, te_labels = _run_epoch(model, te_loader, optimizer, criterion, False)
    te_preds = (te_probs > 0.5).astype(int)

    print(f"\n{'='*40}")
    print(f"Test Accuracy : {te_acc:.4f}")
    print(f"F1 Score      : {f1_score(te_labels, te_preds):.4f}")
    print(f"AUC-ROC       : {roc_auc_score(te_labels, te_probs):.4f}")
    print(f"{'='*40}")

    # Save artifacts
    tok.save(f"{save_dir}/tokenizer.json")
    with open(f"{save_dir}/scaler.pkl",   'wb') as f: pickle.dump(scaler, f)
    with open(f"{save_dir}/meta_cols.json",'w') as f: json.dump(meta_cols, f)
    with open(f"{save_dir}/history.json", 'w') as f: json.dump(history, f)
    np.save(f"{save_dir}/test_probs.npy",  te_probs)
    np.save(f"{save_dir}/test_labels.npy", te_labels)
    print(f"\n✅ All artifacts saved to '{save_dir}/'")


if __name__ == '__main__':
    import sys
    train(sys.argv[1] if len(sys.argv) > 1 else 'data/youtube')