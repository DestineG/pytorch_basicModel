# chapter5-3-GNMT/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class LuongAttention(nn.Module):
    """Multiplicative (dot) attention used by GNMT."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear_in = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, query, keys, values, mask=None):
        """
        Args:
            query: (batch, hidden) decoder hidden state.
            keys: (batch, src_len, hidden) encoder outputs.
            values: (batch, src_len, hidden) usually same as keys.
            mask: (batch, src_len) with True on padded positions.
        Returns:
            context: (batch, hidden)
            attn_weights: (batch, src_len)
        """
        # (batch, hidden) -> (batch, hidden)
        proj_query = self.linear_in(query)
        # (batch, src_len, hidden) bmm (batch, hidden, 1) -> (batch, src_len)
        scores = torch.bmm(keys, proj_query.unsqueeze(2)).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        # (batch, 1, src_len) bmm (batch, src_len, hidden) -> (batch, 1, hidden)
        context = torch.bmm(attn_weights.unsqueeze(1), values).squeeze(1)
        return context, attn_weights


class GNMTEncoder(nn.Module):
    """
    GNMT encoder: first layer is bidirectional, upper layers are unidirectional
    with residual connections starting at the third layer (index >= 2).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.2,
        pad_id: int = 0,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=pad_id)
        self.layers = nn.ModuleList()
        # First layer: bidirectional
        self.layers.append(
            nn.LSTM(
                input_size=embed_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
        )
        # Projection layer to reduce bidirectional output to hidden_size
        self.projection = nn.Linear(2 * hidden_size, hidden_size)
        # Upper layers: unidirectional
        for _ in range(1, num_layers):
            self.layers.append(
                nn.LSTM(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=False,
                )
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_tokens, lengths):
        """
        Args:
            src_tokens: (batch, src_len)
            lengths: (batch,) actual lengths before padding
        Returns:
            outputs: (batch, src_len, hidden)
            hidden: (num_layers, batch, hidden)
            cell: (num_layers, batch, hidden)
        """
        max_src_len = src_tokens.size(1)
        embed = self.embed(src_tokens)
        # 提取每个时间步的有效长度
        # (batch, max_src_len, embed) -> PackedSequence
        packed = nn.utils.rnn.pack_padded_sequence(
            embed, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        layer_outputs = []
        hidden_states = []
        cell_states = []
        residual_input = None
        for idx, layer in enumerate(self.layers):
            # 输出(PackedSequence), 隐藏状态, 记忆状态
            packed_out, (h_n, c_n) = layer(packed)
            # 恢复(pad)为统一长度的tensor
            out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=max_src_len
            )
            # Project bidirectional output to hidden_size after first layer
            if idx == 0:
                out = self.projection(out)
                # Collapse bidirectional hidden of first layer by sum
                h_n = h_n.view(2, h_n.size(1), h_n.size(2))
                c_n = c_n.view(2, c_n.size(1), c_n.size(2))
                h_n = h_n.sum(0, keepdim=True)
                c_n = c_n.sum(0, keepdim=True)
            # 第二层及以后添加残差
            if residual_input is not None and idx >= 2:
                out = out + residual_input
            out = self.dropout(out) if idx + 1 < self.num_layers else out
            residual_input = out
            # 提取每个时间步的有效长度
            packed = nn.utils.rnn.pack_padded_sequence(
                out, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

            hidden_states.append(h_n)
            cell_states.append(c_n)

        outputs = residual_input
        hidden = torch.cat(hidden_states, dim=0)
        cell = torch.cat(cell_states, dim=0)
        return outputs, (hidden, cell)


class GNMTDecoder(nn.Module):
    """
    GNMT decoder with Luong attention and residual connections from layer 2.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.2,
        pad_id: int = 0,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=pad_id)
        self.layers = nn.ModuleList()
        # First layer consumes embedding + context
        self.layers.append(
            nn.LSTMCell(embed_size + hidden_size, hidden_size)
        )
        for _ in range(1, num_layers):
            self.layers.append(nn.LSTMCell(hidden_size, hidden_size))
        self.attn = LuongAttention(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, tgt_tokens, encoder_outputs, src_mask, initial_state=None):
        """
        Args:
            tgt_tokens: (batch, tgt_len) gold tokens (including BOS, excluding EOS)
            encoder_outputs: (batch, src_len, hidden)
            src_mask: (batch, src_len) bool mask with True on padding
            initial_state: tuple of (h, c) each (num_layers, batch, hidden)
        Returns:
            logits: (batch, tgt_len, vocab)
            attn_weights: (batch, tgt_len, src_len)
        """
        batch_size, tgt_len = tgt_tokens.size()
        device = tgt_tokens.device
        if initial_state is None:
            h = [
                torch.zeros(batch_size, self.hidden_size, device=device)
                for _ in range(self.num_layers)
            ]
            c = [
                torch.zeros(batch_size, self.hidden_size, device=device)
                for _ in range(self.num_layers)
            ]
        else:
            h, c = initial_state
            h = [h[i] for i in range(self.num_layers)]
            c = [c[i] for i in range(self.num_layers)]

        # (batch, max_tgt_len, embed)
        embeddings = self.embed(tgt_tokens)
        context = torch.zeros(batch_size, self.hidden_size, device=device)
        outputs = []
        attn_scores = []

        for t in range(tgt_len):
            # (batch, embed) 第一个时间步都是BOS
            # 训练时使用teacher forcing，直接使用目标词的embedding
            # 推理时使用上一个时间步预测的词的embedding
            emb_t = embeddings[:, t, :]
            # (batch, embed + hidden)
            layer_input = torch.cat([emb_t, context], dim=-1)
            new_h = []
            new_c = []
            # 逐层LSTMCell前向传播
            for idx, cell in enumerate(self.layers):
                # Save input for residual connection (layers >= 2)
                residual = layer_input if idx >= 2 else None
                # (batch, embed + hidden) -> (batch, hidden), (batch, hidden)
                h_t, c_t = cell(layer_input, (h[idx], c[idx]))
                if residual is not None:
                    h_t = h_t + residual
                layer_input = self.dropout(h_t) if idx + 1 < self.num_layers else h_t
                new_h.append(h_t)
                new_c.append(c_t)
            h, c = new_h, new_c

            # (batch, hidden), (batch, max_src_len)
            context, attn = self.attn(
                h[-1],                  # Q: (batch, hidden)
                encoder_outputs,        # K: (batch, max_src_len, hidden)
                encoder_outputs,        # V: (batch, max_src_len, hidden)
                src_mask                # mask: (batch, max_src_len)
            )
            attn_scores.append(attn)
            # (batch, 2 * hidden) -> (batch, vocab), 用当前词和源语言语句的特征池预测下一个词
            logits = self.out_proj(torch.cat([h[-1], context], dim=-1))
            outputs.append(logits)

        logits = torch.stack(outputs, dim=1)
        attn_scores = torch.stack(attn_scores, dim=1)
        return logits, attn_scores


class GNMT(nn.Module):
    """Full GNMT model wrapping encoder and decoder."""

    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        embed_size: int = 512,
        hidden_size: int = 512,
        num_layers: int = 4,
        dropout: float = 0.2,
        pad_id: int = 0,
    ):
        super().__init__()
        # 源语言特征池提取器
        self.encoder = GNMTEncoder(
            vocab_size=src_vocab,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            pad_id=pad_id,
        )
        # 自回归预测器(encoder生成的特征池为decoder指定预测方向)
        self.decoder = GNMTDecoder(
            vocab_size=tgt_vocab,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            pad_id=pad_id,
        )

    def forward(self, src_tokens, src_lengths, tgt_tokens):
        """
        Args:
            src_tokens: (batch, src_len)
            src_lengths: (batch,)
            tgt_tokens: (batch, tgt_len)
        Returns:
            logits: (batch, tgt_len, tgt_vocab)
            attn: (batch, tgt_len, src_len)
        """
        # (B, max_src_len, embed) -> (B, max_src_len, hidden)
        enc_out, enc_state = self.encoder(src_tokens, src_lengths)
        # Build source mask: True for padding positions
        max_src_len = src_tokens.size(1)
        device = src_tokens.device
        range_row = torch.arange(max_src_len, device=device).unsqueeze(0)
        src_mask = range_row >= src_lengths.unsqueeze(1)
        logits, attn = self.decoder(tgt_tokens, enc_out, src_mask, enc_state)
        return logits, attn


if __name__ == "__main__":
    x = torch.tensor([[1, 2, 3, 4, 0], [5, 6, 7, 0, 0]])
    lengths = torch.tensor([4, 3])
    y = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    model = GNMT(src_vocab=10, tgt_vocab=10, num_layers=4)
    logits, attn = model(x, lengths, y)
    print("Logits shape:", logits.shape)  # (batch, tgt_len, tgt_vocab)
    print("Attention shape:", attn.shape)  # (batch, tgt_len, src_len)