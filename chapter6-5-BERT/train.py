# chapter6-5-BERT/train.py

from .data import build_dataloader
from .model import BERTModel, BERTModelConfig


def train():
    dataloader = build_dataloader(
        min_freq=10,
        max_size=30522,
        max_seq_len=128,
        mlm_prob=0.15,
        batch_size=32,
        num_workers=0
)

    config = BERTModelConfig()
    model = BERTModel(config)

    for batch in dataloader:
        input_ids = batch["input_ids"]
        token_type_ids = batch["token_type_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        next_sentence_label = batch["next_sentence_label"]
        outputs = model(input_ids, token_type_ids, attention_mask)

        print("Input shape:", input_ids.shape)
        print("Token type IDs shape:", token_type_ids.shape)
        print("Attention mask shape:", attention_mask.shape)
        print("Labels shape:", labels.shape)
        print("Next sentence labels shape:", next_sentence_label.shape)

        print(outputs["mlm_logits"].shape)
        print(outputs["nsp_logits"].shape)
        break

# [Counter] Loading cached counter from token_counter.pkl
# Input shape: torch.Size([32, 128])
# Token type IDs shape: torch.Size([32, 128])
# Attention mask shape: torch.Size([32, 128])
# Labels shape: torch.Size([32, 128])
# Next sentence labels shape: torch.Size([32])
# torch.Size([32, 128, 30522])
# torch.Size([32, 2])

if __name__ == "__main__":
    train()