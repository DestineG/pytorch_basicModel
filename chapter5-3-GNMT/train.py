"""
Simple training script for GNMT on the zh-en CSV corpus.
Usage (example):
    python train.py --csv G:\\datasets\\wmt_zh_en_training_corpus.csv --epochs 1
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from data import create_dataloader, PAD_ID
from model import GNMT


def parse_args():
    parser = argparse.ArgumentParser(description="Train GNMT on zh-en corpus.")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV dataset.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--max-vocab", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--clip", type=float, default=1.0, help="Grad clipping max norm.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save", type=str, default=None, help="Path to save model checkpoint.")
    return parser.parse_args()


def train_one_epoch(model, dataloader, optimizer, criterion, device, clip_norm, log_interval):
    model.train()
    total_loss = 0.0
    tokens_seen = 0
    for step, (src, src_len, tgt) in enumerate(dataloader, start=1):
        src = src.to(device)
        src_len = src_len.to(device)
        tgt = tgt.to(device)

        # Shifted target for teacher forcing
        decoder_in = tgt[:, :-1]
        target = tgt[:, 1:]

        logits, _ = model(src, src_len, decoder_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        if clip_norm is not None and clip_norm > 0:
            clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        total_loss += loss.item() * target.numel()
        tokens_seen += target.numel()

        if step % log_interval == 0:
            ppl = math.exp(total_loss / tokens_seen) if tokens_seen > 0 else float("inf")
            print(f"Step {step}: loss={total_loss / tokens_seen:.4f}, ppl={ppl:.2f}")

    avg_loss = total_loss / max(1, tokens_seen)
    return avg_loss


def main():
    args = parse_args()
    device = torch.device(args.device)

    dataloader, dataset = create_dataloader(
        csv_path=args.csv,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        min_freq=args.min_freq,
        max_size=args.max_vocab,
    )

    vocab_size = len(dataset.vocab)
    model = GNMT(
        src_vocab=vocab_size,
        tgt_vocab=vocab_size,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        dropout=args.dropout,
        pad_id=PAD_ID,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = Adam(model.parameters(), lr=args.lr)

    print(f"Device: {device}")
    print(f"Vocab size: {vocab_size}")
    print(f"Training samples: {len(dataset)}")

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(
            model,
            dataloader,
            optimizer,
            criterion,
            device,
            args.clip,
            args.log_interval,
        )
        ppl = math.exp(avg_loss)
        print(f"Epoch {epoch} done. avg_loss={avg_loss:.4f}, ppl={ppl:.2f}")

    if args.save:
        ckpt_path = Path(args.save)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "vocab": dataset.vocab.stoi,
                "args": vars(args),
            },
            ckpt_path,
        )
        print(f"Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()

