# chapter6-5-BERT/embedding.py

from dataclasses import dataclass

import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, padding_idx: int):
        super(TokenEmbedding, self).__init__()
        self.net = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class PositionEmbedding(nn.Module):
    def __init__(self, max_len: int, hidden_size: int):
        super(PositionEmbedding, self).__init__()
        self.net = nn.Embedding(max_len, hidden_size)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.size()
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        return self.net(positions)

class SegmentEmbedding(nn.Module):
    def __init__(self, type_vocab_size: int, hidden_size: int):
        super(SegmentEmbedding, self).__init__()
        self.net = nn.Embedding(type_vocab_size, hidden_size)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

@dataclass
class BERTEmbeddingConfig:
    vocab_size: int = 30522
    type_vocab_size: int = 2
    max_len: int = 512
    hidden_size: int = 768
    padding_idx: int = 0
    hidden_dropout_prob: float = 0.1
class BERTEmbedding(nn.Module):
    def __init__(self, config: BERTEmbeddingConfig):
        super(BERTEmbedding, self).__init__()
        self.token_embedding = TokenEmbedding(config.vocab_size, config.hidden_size, config.padding_idx)
        self.position_embedding = PositionEmbedding(config.max_len, config.hidden_size)
        self.segment_embedding = SegmentEmbedding(config.type_vocab_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # (batch_size, seq_len), (batch_size, seq_len), (batch_size, seq_len)
    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor=None, token_type_ids: torch.Tensor=None) -> torch.Tensor:
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
        
        # (batch_size, seq_len) -> (batch_size, seq_len, hidden_size)
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        segment_embeds = self.segment_embedding(token_type_ids)

        # (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, hidden_size)
        embeddings = token_embeds + position_embeds + segment_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

