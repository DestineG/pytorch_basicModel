# chapter6-5-BERT/encoder.py

from dataclasses import dataclass
import math
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super(MultiHeadSelfAttention, self).__init__()
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.out = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.size()
        x = x.view(B, L, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)
    
    # (batch_size, seq_len, hidden_size), (batch_size, seq_len)
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor=None, head_mask: torch.Tensor=None) -> torch.Tensor:

        # (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, hidden_size)
        Q = self.query(hidden_states)
        K = self.key(hidden_states)
        V = self.value(hidden_states)

        # (batch_size, seq_len, hidden_size) -> (batch_size, num_heads, seq_len, head_dim)
        Q = self.transpose_for_scores(Q)
        K = self.transpose_for_scores(K)
        V = self.transpose_for_scores(V)

        # (batch_size, num_heads, seq_len, head_dim) x (batch_size, num_heads, head_dim, seq_len)
        # -> (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            # (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # (batch_size, num_heads, seq_len, seq_len) -> (batch_size, num_heads, seq_len, seq_len)
        attn_weights = torch.softmax(scores, dim=-1)
        if head_mask is not None:
            # (num_heads,) -> (1, num_heads, 1, 1)
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            # (batch_size, num_heads) -> (batch_size, num_heads, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.view(head_mask.size(0), head_mask.size(1), 1, 1)
            attn_weights = attn_weights * head_mask
        attn_weights = self.dropout(attn_weights)

        # (batch_size, num_heads, seq_len, seq_len) x (batch_size, num_heads, seq_len, head_dim)
        # -> (batch_size, num_heads, seq_len, head_dim)
        # -> (batch_size, seq_len, num_heads, head_dim)
        # -> (batch_size, seq_len, hidden_size)
        context = torch.matmul(attn_weights, V)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(hidden_states.size(0), hidden_states.size(1), self.hidden_size)

        # (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, hidden_size)
        output = self.out(context)

        # (batch_size, seq_len, hidden_size), (batch_size, num_heads, seq_len, seq_len)
        return output, attn_weights

class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, dim_feedforward: int, dropout: float):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, hidden_size),
            nn.Dropout(dropout)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class BERTEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super(BERTEncoderLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, nhead, dropout)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-12)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
    
    # (batch_size, seq_len, hidden_size), (batch_size, seq_len), (num_heads,)
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor=None, head_mask: torch.Tensor=None):
        # (batch_size, seq_len, hidden_size), (batch_size, seq_len), (num_heads,) 
        # -> (batch_size, seq_len, hidden_size), (batch_size, num_heads, seq_len, seq_len)
        attn_output, attn_weights = self.self_attn(hidden_states, attention_mask, head_mask)

        # (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, hidden_size)
        hidden_states = self.norm1(hidden_states + self.dropout(attn_output))

        # (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, hidden_size)
        ffn_output = self.feed_forward(hidden_states)

        # (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, hidden_size)
        hidden_states = self.norm2(hidden_states + self.dropout(ffn_output))

        # (batch_size, seq_len, hidden_size), (batch_size, num_heads, seq_len, seq_len)
        return hidden_states, attn_weights

@dataclass
class BERTEncoderConfig:
    hidden_size: int = 768
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    num_hidden_layers: int = 12
    output_attentions: bool = False
    output_hidden_states: bool = False
class BERTEncoder(nn.Module):
    def __init__(self, config: BERTEncoderConfig):
        super(BERTEncoder, self).__init__()
        self.config = config
        self.output_attentions = self.config.output_attentions
        self.output_hidden_states = self.config.output_hidden_states
        self.layer = nn.ModuleList([BERTEncoderLayer(
                                        d_model=self.config.hidden_size,
                                        nhead=self.config.num_attention_heads,
                                        dim_feedforward=self.config.intermediate_size,
                                        dropout=self.config.hidden_dropout_prob) \
                                    for _ in range(self.config.num_hidden_layers)])
 
    # (batch_size, seq_len, hidden_size), (batch_size, seq_len), (num_layers, num_heads)
    # 中间状态，注意力掩码，头掩码
    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        # 迭代计算中间的输出
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = None
            if head_mask is not None:
                layer_head_mask = head_mask[i]

            # (batch_size, seq_len, hidden_size), (batch_size, seq_len), (num_heads,)
            # -> (batch_size, seq_len, hidden_size), (batch_size, num_heads, seq_len, seq_len)
            layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        # 包含最后的输出、中间的输出，以及自注意力的权重
        return outputs  