# chapter7-3-Tacotron2/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .data import TextProcessor, build_dataloader


class Encoder(nn.Module):
    """Tacotron2 Encoder: Character embedding + 3 Conv layers + Bidirectional LSTM"""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        encoder_conv_filters: int = 512,
        encoder_conv_kernel_size: int = 5,
        encoder_lstm_units: int = 256,
        dropout: float = 0.5,
        pad_id: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        
        # 3个卷积层，每个后面跟BatchNorm和ReLU
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, encoder_conv_filters, 
                         kernel_size=encoder_conv_kernel_size, 
                         padding=(encoder_conv_kernel_size - 1) // 2),
                nn.BatchNorm1d(encoder_conv_filters),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            nn.Sequential(
                nn.Conv1d(encoder_conv_filters, encoder_conv_filters, 
                         kernel_size=encoder_conv_kernel_size, 
                         padding=(encoder_conv_kernel_size - 1) // 2),
                nn.BatchNorm1d(encoder_conv_filters),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            nn.Sequential(
                nn.Conv1d(encoder_conv_filters, encoder_conv_filters, 
                         kernel_size=encoder_conv_kernel_size, 
                         padding=(encoder_conv_kernel_size - 1) // 2),
                nn.BatchNorm1d(encoder_conv_filters),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        ])
        
        # 双向LSTM
        self.lstm = nn.LSTM(
            encoder_conv_filters,
            encoder_lstm_units,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
    def forward(self, text_ids, text_lengths):
        """
        Args:
            text_ids: (B, T_text) - 文本ID序列
            text_lengths: (B,) - 文本长度
        Returns:
            encoder_outputs: (B, T_text, encoder_lstm_units * 2)
        """
        # Embedding: (B, T_text) -> (B, T_text, embed_dim)
        embedded = self.embedding(text_ids)
        
        # Conv1d需要(B, C, T)，所以转置
        # (B, T_text, embed_dim) -> (B, embed_dim, T_text)
        x = embedded.transpose(1, 2)
        
        # 通过3个卷积层
        for conv in self.conv_layers:
            x = conv(x)
        
        # 转回 (B, T_text, C) 用于LSTM
        x = x.transpose(1, 2)
        
        # Pack padded sequence for LSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            x, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # LSTM
        outputs, _ = self.lstm(packed)
        
        # Unpack
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True
        )
        
        return outputs


class LocationSensitiveAttention(nn.Module):
    """Location-Sensitive Attention mechanism for Tacotron2"""
    
    def __init__(
        self,
        attention_dim: int = 128,
        attention_rnn_dim: int = 1024,
        encoder_dim: int = 512,
        num_filters: int = 32,
        kernel_size: int = 31,
    ):
        super().__init__()
        self.attention_dim = attention_dim
        self.attention_rnn_dim = attention_rnn_dim
        self.encoder_dim = encoder_dim
        
        # Location-sensitive attention uses convolution on previous attention weights
        self.location_conv = nn.Conv1d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False
        )
        self.location_projection = nn.Linear(num_filters, attention_dim, bias=False)
        
        # Query projection (from decoder RNN output)
        self.query_projection = nn.Linear(attention_rnn_dim, attention_dim, bias=False)
        
        # Key projection (from encoder outputs)
        self.key_projection = nn.Linear(encoder_dim, attention_dim, bias=False)
        
        # Value projection
        self.value_projection = nn.Linear(encoder_dim, encoder_dim, bias=False)
        
        # Attention weights projection
        self.attention_weights_projection = nn.Linear(attention_dim, 1, bias=False)
        
    def forward(self, query, keys, values, attention_weights_cum, mask=None):
        """
        Args:
            query: (B, attention_rnn_dim) - decoder RNN output
            keys: (B, T_text, encoder_dim) - encoder outputs
            values: (B, T_text, encoder_dim) - encoder outputs (usually same as keys)
            attention_weights_cum: (B, T_text) - cumulative attention weights from previous step
            mask: (B, T_text) - True for padding positions
        Returns:
            context: (B, encoder_dim)
            attention_weights: (B, T_text)
        """
        batch_size = query.size(0)
        seq_len = keys.size(1)
        
        # Process location features from previous attention weights
        # (B, T_text) -> (B, 1, T_text) -> (B, num_filters, T_text)
        location_features = self.location_conv(attention_weights_cum.unsqueeze(1))
        # (B, num_filters, T_text) -> (B, T_text, num_filters) -> (B, T_text, attention_dim)
        location_features = self.location_projection(location_features.transpose(1, 2))
        
        # Project query: (B, attention_rnn_dim) -> (B, attention_dim)
        query_proj = self.query_projection(query)
        
        # Project keys: (B, T_text, encoder_dim) -> (B, T_text, attention_dim)
        keys_proj = self.key_projection(keys)
        
        # Compute attention scores
        # (B, 1, attention_dim) + (B, T_text, attention_dim) + (B, T_text, attention_dim)
        # -> (B, T_text, attention_dim)
        attention_scores = torch.tanh(
            query_proj.unsqueeze(1) + keys_proj + location_features
        )
        
        # (B, T_text, attention_dim) -> (B, T_text, 1) -> (B, T_text)
        attention_scores = self.attention_weights_projection(attention_scores).squeeze(-1)
        
        # Apply mask
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask, float("-inf"))
        
        # Softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Compute context: (B, 1, T_text) @ (B, T_text, encoder_dim) -> (B, 1, encoder_dim) -> (B, encoder_dim)
        values_proj = self.value_projection(values)
        context = torch.bmm(attention_weights.unsqueeze(1), values_proj).squeeze(1)
        
        return context, attention_weights


class Prenet(nn.Module):
    """Pre-net for decoder input processing"""
    
    def __init__(
        self,
        in_dim: int,
        prenet_dim: int = 256,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(in_dim, prenet_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(prenet_dim, prenet_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ])
    
    def forward(self, x):
        """
        Args:
            x: (B, in_dim) or (B, T, in_dim)
        Returns:
            output: same shape as input
        """
        for layer in self.layers:
            x = layer(x)
        return x


class Postnet(nn.Module):
    """Post-net for mel-spectrogram refinement"""
    
    def __init__(
        self,
        n_mels: int = 80,
        postnet_dim: int = 512,
        postnet_kernel_size: int = 5,
        num_layers: int = 5,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(
            nn.Sequential(
                nn.Conv1d(n_mels, postnet_dim, kernel_size=postnet_kernel_size, 
                         padding=(postnet_kernel_size - 1) // 2),
                nn.BatchNorm1d(postnet_dim),
                nn.Tanh(),
                nn.Dropout(dropout)
            )
        )
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(postnet_dim, postnet_dim, kernel_size=postnet_kernel_size,
                             padding=(postnet_kernel_size - 1) // 2),
                    nn.BatchNorm1d(postnet_dim),
                    nn.Tanh(),
                    nn.Dropout(dropout)
                )
            )
        
        # Last layer (no activation, no dropout)
        self.layers.append(
            nn.Sequential(
                nn.Conv1d(postnet_dim, n_mels, kernel_size=postnet_kernel_size,
                         padding=(postnet_kernel_size - 1) // 2),
                nn.BatchNorm1d(n_mels)
            )
        )
    
    def forward(self, mel_outputs):
        """
        Args:
            mel_outputs: (B, n_mels, T) - mel spectrogram
        Returns:
            refined: (B, n_mels, T) - refined mel spectrogram
        """
        x = mel_outputs
        for layer in self.layers:
            x = layer(x)
        return x


class Decoder(nn.Module):
    """Tacotron2 Decoder"""
    
    def __init__(
        self,
        n_mels: int = 80,
        encoder_dim: int = 512,
        prenet_dim: int = 256,
        attention_rnn_dim: int = 1024,
        decoder_rnn_dim: int = 1024,
        attention_dim: int = 128,
        attention_location_n_filters: int = 32,
        attention_location_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.encoder_dim = encoder_dim
        self.attention_rnn_dim = attention_rnn_dim
        self.decoder_rnn_dim = decoder_rnn_dim
        
        # Pre-net
        self.prenet = Prenet(n_mels, prenet_dim, dropout=0.5)
        
        # Attention RNN
        self.attention_rnn = nn.LSTMCell(
            prenet_dim + encoder_dim,  # input: prenet output + context
            attention_rnn_dim
        )
        
        # Attention mechanism
        self.attention = LocationSensitiveAttention(
            attention_dim=attention_dim,
            attention_rnn_dim=attention_rnn_dim,
            encoder_dim=encoder_dim,
            num_filters=attention_location_n_filters,
            kernel_size=attention_location_kernel_size,
        )
        
        # Decoder RNN
        self.decoder_rnn = nn.LSTMCell(
            attention_rnn_dim + encoder_dim,  # input: attention_rnn output + context
            decoder_rnn_dim
        )
        
        # Linear projections
        self.linear_projection = nn.Linear(
            decoder_rnn_dim + encoder_dim,  # decoder_rnn output + context
            n_mels
        )
        
        self.gate_layer = nn.Linear(
            decoder_rnn_dim + encoder_dim,  # decoder_rnn output + context
            1
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, encoder_outputs, mel_targets, text_lengths, mask=None):
        """
        Args:
            encoder_outputs: (B, T_text, encoder_dim) - encoder outputs
            mel_targets: (B, n_mels, T_mel) - target mel spectrogram (for teacher forcing)
            text_lengths: (B,) - text lengths for masking
            mask: (B, T_text) - True for padding positions
        Returns:
            mel_outputs: (B, n_mels, T_mel) - predicted mel spectrogram
            gate_outputs: (B, T_mel) - gate predictions
            alignments: (B, T_mel, T_text) - attention alignments
        """
        batch_size = encoder_outputs.size(0)
        max_mel_len = mel_targets.size(2) if mel_targets is not None else 1000
        
        # Initialize states
        attention_rnn_hidden = torch.zeros(
            batch_size, self.attention_rnn_dim, device=encoder_outputs.device
        )
        attention_rnn_cell = torch.zeros(
            batch_size, self.attention_rnn_dim, device=encoder_outputs.device
        )
        decoder_rnn_hidden = torch.zeros(
            batch_size, self.decoder_rnn_dim, device=encoder_outputs.device
        )
        decoder_rnn_cell = torch.zeros(
            batch_size, self.decoder_rnn_dim, device=encoder_outputs.device
        )
        
        # Initialize attention weights
        attention_weights = torch.zeros(
            batch_size, encoder_outputs.size(1), device=encoder_outputs.device
        )
        attention_weights_cum = torch.zeros(
            batch_size, encoder_outputs.size(1), device=encoder_outputs.device
        )
        
        # Create mask if not provided
        if mask is None:
            max_text_len = encoder_outputs.size(1)
            device = encoder_outputs.device
            range_row = torch.arange(max_text_len, device=device).unsqueeze(0)
            mask = range_row >= text_lengths.unsqueeze(1)
        
        # First decoder input (go frame)
        decoder_input = torch.zeros(
            batch_size, self.n_mels, device=encoder_outputs.device
        )
        
        # Initialize context (use mean of encoder outputs)
        context = encoder_outputs.mean(dim=1)  # (B, encoder_dim)
        
        mel_outputs = []
        gate_outputs = []
        alignments = []
        
        for t in range(max_mel_len):
            # Pre-net
            prenet_output = self.prenet(decoder_input)  # (B, prenet_dim)
            
            # Attention RNN
            attention_rnn_input = torch.cat([prenet_output, context], dim=-1)
            attention_rnn_hidden, attention_rnn_cell = self.attention_rnn(
                attention_rnn_input,
                (attention_rnn_hidden, attention_rnn_cell)
            )
            attention_rnn_hidden = self.dropout(attention_rnn_hidden)
            
            # Attention
            context, attention_weights = self.attention(
                attention_rnn_hidden,
                encoder_outputs,
                encoder_outputs,
                attention_weights_cum,
                mask
            )
            attention_weights_cum = attention_weights_cum + attention_weights
            alignments.append(attention_weights)
            
            # Decoder RNN
            decoder_rnn_input = torch.cat([attention_rnn_hidden, context], dim=-1)
            decoder_rnn_hidden, decoder_rnn_cell = self.decoder_rnn(
                decoder_rnn_input,
                (decoder_rnn_hidden, decoder_rnn_cell)
            )
            decoder_rnn_hidden = self.dropout(decoder_rnn_hidden)
            
            # Linear projections
            decoder_output = torch.cat([decoder_rnn_hidden, context], dim=-1)
            mel_output = self.linear_projection(decoder_output)  # (B, n_mels)
            gate_output = self.gate_layer(decoder_output)  # (B, 1)
            
            mel_outputs.append(mel_output)
            gate_outputs.append(gate_output.squeeze(-1))
            
            # Teacher forcing: use ground truth mel frame
            if mel_targets is not None and self.training:
                decoder_input = mel_targets[:, :, t]  # (B, n_mels)
            else:
                decoder_input = mel_output
        
        # Stack outputs
        mel_outputs = torch.stack(mel_outputs, dim=2)  # (B, n_mels, T_mel)
        gate_outputs = torch.stack(gate_outputs, dim=1)  # (B, T_mel)
        alignments = torch.stack(alignments, dim=1)  # (B, T_mel, T_text)
        
        return mel_outputs, gate_outputs, alignments


# cursor 生成
class Tacotron2(nn.Module):
    """Complete Tacotron2 model"""
    
    def __init__(
        self,
        vocab_size: int,
        n_mels: int = 80,
        embed_dim: int = 512,
        encoder_conv_filters: int = 512,
        encoder_conv_kernel_size: int = 5,
        encoder_lstm_units: int = 256,
        prenet_dim: int = 256,
        attention_rnn_dim: int = 1024,
        decoder_rnn_dim: int = 1024,
        attention_dim: int = 128,
        attention_location_n_filters: int = 32,
        attention_location_kernel_size: int = 31,
        postnet_dim: int = 512,
        postnet_kernel_size: int = 5,
        postnet_num_layers: int = 5,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        
        self.encoder = Encoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            encoder_conv_filters=encoder_conv_filters,
            encoder_conv_kernel_size=encoder_conv_kernel_size,
            encoder_lstm_units=encoder_lstm_units,
            dropout=dropout,
            pad_id=pad_id,
        )
        
        encoder_dim = encoder_lstm_units * 2  # bidirectional
        
        self.decoder = Decoder(
            n_mels=n_mels,
            encoder_dim=encoder_dim,
            prenet_dim=prenet_dim,
            attention_rnn_dim=attention_rnn_dim,
            decoder_rnn_dim=decoder_rnn_dim,
            attention_dim=attention_dim,
            attention_location_n_filters=attention_location_n_filters,
            attention_location_kernel_size=attention_location_kernel_size,
            dropout=dropout,
        )
        
        self.postnet = Postnet(
            n_mels=n_mels,
            postnet_dim=postnet_dim,
            postnet_kernel_size=postnet_kernel_size,
            num_layers=postnet_num_layers,
            dropout=dropout,
        )
    
    def forward(self, text_ids, text_lengths, mel_targets=None):
        """
        Args:
            text_ids: (B, T_text) - 文本ID序列
            text_lengths: (B,) - 文本长度
            mel_targets: (B, n_mels, T_mel) - 目标mel频谱图 (训练时使用，推理时为None)
        Returns:
            mel_outputs: (B, n_mels, T_mel) - 预测的mel频谱图
            mel_outputs_postnet: (B, n_mels, T_mel) - 经过postnet的mel频谱图
            gate_outputs: (B, T_mel) - gate预测
            alignments: (B, T_mel, T_text) - 注意力对齐
        """
        # Encoder
        encoder_outputs = self.encoder(text_ids, text_lengths)
        
        # Decoder
        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mel_targets, text_lengths
        )
        
        # Post-net
        mel_outputs_postnet = self.postnet(mel_outputs) + mel_outputs
        
        return mel_outputs, mel_outputs_postnet, gate_outputs, alignments


if __name__ == "__main__":
    # 测试模型
    batch_size = 4
    vocab_size = TextProcessor.vocab_size
    n_mels = 80
    
    # 创建模型
    model = Tacotron2(vocab_size=vocab_size, n_mels=n_mels)
    
    # 创建示例输入
    dataloader = build_dataloader(batch_size=batch_size, shuffle=False)
    batch = next(iter(dataloader))
    text_ids, text_lengths, mel_targets, _ = batch
    
    # 前向传播
    mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model(
        text_ids, text_lengths, mel_targets
    )
    
    print("Model input shapes:")
    print(f"text_ids: {text_ids.shape}")  # (B, T_text)
    print(f"text_lengths: {text_lengths.shape}")  # (B,)
    print(f"mel_targets: {mel_targets.shape}")  # (B, n_mels, T_mel)

    print("Model output shapes:")
    print(f"mel_outputs: {mel_outputs.shape}")  # (B, n_mels, T_mel)
    print(f"mel_outputs_postnet: {mel_outputs_postnet.shape}")  # (B, n_mels, T_mel)
    print(f"gate_outputs: {gate_outputs.shape}")  # (B, T_mel)
    print(f"alignments: {alignments.shape}")  # (B, T_mel, T_text)

# Model input shapes:
# text_ids: torch.Size([4, 156])
# text_lengths: torch.Size([4])
# mel_targets: torch.Size([4, 80, 833])

# Model output shapes:
# mel_outputs: torch.Size([4, 80, 833])
# mel_outputs_postnet: torch.Size([4, 80, 833])
# gate_outputs: torch.Size([4, 833])
# alignments: torch.Size([4, 833, 156])