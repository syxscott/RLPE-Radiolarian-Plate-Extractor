from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rlpe.association import PanelLabelSpeciesMatcher


class MatchDataset(Dataset):
    def __init__(self, jsonl_path: Path):
        self.rows = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        panel_feats = torch.tensor(row["panel_features"], dtype=torch.float32)
        label_feats = torch.tensor(row["label_features"], dtype=torch.float32)
        species_feats = torch.tensor(row["species_features"], dtype=torch.float32)
        y_pl = torch.tensor(row["target_panel_label"], dtype=torch.long)
        y_ps = torch.tensor(row["target_panel_species"], dtype=torch.long)
        return panel_feats, label_feats, species_feats, y_pl, y_ps


def train(args) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = MatchDataset(Path(args.train_jsonl))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    model = PanelLabelSpeciesMatcher(feature_dim=args.feature_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        loss_sum = 0.0
        for p, l, s, y_pl, y_ps in loader:
            p, l, s, y_pl, y_ps = p.to(device), l.to(device), s.to(device), y_pl.to(device), y_ps.to(device)
            logits_pl, logits_ps = model(p, l, s)
            loss = ce(logits_pl, y_pl) + ce(logits_ps, y_ps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
        print(f"epoch={epoch+1} loss={loss_sum/max(1, len(loader)):.4f}")

    output = Path(args.output_ckpt)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "feature_dim": args.feature_dim, "hidden_dim": args.hidden_dim}, output)
    print(f"saved={output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Train RLPE neural graph matcher.")
    parser.add_argument("--train-jsonl", type=str, required=True)
    parser.add_argument("--output-ckpt", type=str, required=True)
    parser.add_argument("--feature-dim", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
