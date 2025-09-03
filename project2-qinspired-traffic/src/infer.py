import argparse, torch
from datasets import toy_classification_split, toy_regression_split
from models.core import MLP, Regressor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["clf", "reg"], default="clf")
    ap.add_argument("--ckpt", type=str, required=True)
    args = ap.parse_args()

    if args.task == "clf":
        Xtr, ytr, Xte, yte = toy_classification_split()
        model = MLP(d_in=Xtr.shape[1], d_out=2)
    else:
        Xtr, ytr, Xte, yte = toy_regression_split()
        model = Regressor(d_in=Xtr.shape[1])

    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    with torch.no_grad():
        out = model(Xte[:8])
    print("Sample output (first 8 rows):\n", out)

if __name__ == "__main__":
    main()
