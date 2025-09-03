import argparse, os, torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm
from datasets import toy_classification_split, toy_regression_split
from models.core import MLP, Regressor
from pathlib import Path

# Personal note: keep this tiny & friendly. You can swap in real loaders later.
TASKS = {"clf": toy_classification_split, "reg": toy_regression_split}

def train_classification(args):
    Xtr, ytr, Xte, yte = TASKS["clf"](seed=args.seed)
    model = MLP(d_in=Xtr.shape[1], d_hidden=args.hidden, d_out=args.num_classes)
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    best = 0.0

    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        logits = model(Xtr)
        loss = crit(logits, ytr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            acc = (model(Xte).argmax(1) == yte).float().mean().item()
        print(f"[epoch {epoch+1}] loss={loss.item():.4f} acc={acc:.4f}")
        best = max(best, acc)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    with open(out_dir / "metrics.txt", "w") as f: f.write(f"best_acc={best:.4f}\n")
    print(f"Saved to {out_dir}")

def train_regression(args):
    Xtr, ytr, Xte, yte = TASKS["reg"](seed=args.seed)
    model = Regressor(d_in=Xtr.shape[1], d_hidden=args.hidden)
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.MSELoss()
    best = 1e9

    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        pred = model(Xtr)
        loss = crit(pred, ytr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            mse = crit(model(Xte), yte).item()
        print(f"[epoch {epoch+1}] loss={loss.item():.4f} mse={mse:.4f}")
        best = min(best, mse)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    with open(out_dir / "metrics.txt", "w") as f: f.write(f"best_mse={best:.4f}\n")
    print(f"Saved to {out_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["clf", "reg"], default="clf")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="runs/base")
    args = ap.parse_args()

    if args.task == "clf":
        train_classification(args)
    else:
        train_regression(args)

if __name__ == "__main__":
    main()
