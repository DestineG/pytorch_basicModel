# chapter6-5-BERT/model.py

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .embedding import BERTEmbedding, BERTEmbeddingConfig
from .encoder import BERTEncoder, BERTEncoderConfig


class MLMHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        """
        hidden_states: [B, L, H]
        return: logits [B, L, vocab_size]
        """
        x = self.dense(hidden_states)
        x = self.act(x)
        x = self.layer_norm(x)
        logits = self.decoder(x)
        return logits

class NSPHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, cls_hidden: torch.Tensor):
        """
        cls_hidden: [B, H]
        return: logits [B, 2]
        """
        return self.classifier(cls_hidden)

@dataclass
class BERTModelConfig:
    embedding: BERTEmbeddingConfig = field(default_factory=BERTEmbeddingConfig)

    encoder: BERTEncoderConfig = field(default_factory=BERTEncoderConfig)

    output_attentions: bool = False
    output_hidden_states: bool = False

class BERTModel(nn.Module):
    def __init__(self, config: BERTModelConfig):
        super().__init__()
        self.config = config

        config.encoder.output_attentions = config.output_attentions
        config.encoder.output_hidden_states = config.output_hidden_states

        self.embedding = BERTEmbedding(config.embedding)
        self.encoder = BERTEncoder(config.encoder)

        self.mlm_head = MLMHead(
            hidden_size=config.embedding.hidden_size,
            vocab_size=config.embedding.vocab_size,
        )
        self.nsp_head = NSPHead(
            hidden_size=config.embedding.hidden_size
        )
    
    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor=None, attention_mask: torch.Tensor=None) -> torch.Tensor:
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        # (batch_size, seq_len) -> (batch_size, seq_len, hidden_size)
        embeddings = self.embedding(input_ids, token_type_ids)
        # (batch_size, seq_len, hidden_size)
        # -> (batch_size, seq_len, hidden_size)
        # -> (batch_size, layers, seq_len, hidden_size)
        # -> (batch_size, layers, num_heads, seq_len, seq_len)
        encoder_output = self.encoder(embeddings, attention_mask)
        sequence_output = encoder_output[0]
        all_hidden_states = (
            encoder_output[1] if self.config.output_hidden_states else None
        )
        all_attentions = (
            encoder_output[2] if self.config.output_attentions else None
        )

        # MLM
        # (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, vocab_size)
        mlm_logits = self.mlm_head(sequence_output)

        # NSP
        # (batch_size, hidden_size) -> (batch_size, 2)
        cls_output = sequence_output[:, 0]
        nsp_logits = self.nsp_head(cls_output)

        return {
            "mlm_logits": mlm_logits,
            "nsp_logits": nsp_logits,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
        }


if __name__ == "__main__":
    # Example usage
    config = BERTModelConfig()
    model = BERTModel(config)

    input_ids = torch.tensor([[101, 2054, 2003, 1996, 3185, 102], [101, 2129, 2024, 2017, 102, 0]])
    token_type_ids = torch.tensor([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]])

    out = model(input_ids, token_type_ids, attention_mask)
    print(out["mlm_logits"].shape)
    print(out["nsp_logits"].shape)
    print(out["hidden_states"])
    print(out["attentions"])
