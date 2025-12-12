"""
Inference script for GNMT translation model.
Supports greedy decoding and beam search.

Usage:
    # Single sentence
    python infer.py --checkpoint model.pt --text "表演 的 明星 是 X 女孩 团队"
    
    # Batch from file
    python infer.py --checkpoint model.pt --input-file test.txt --output-file result.txt
    
    # With beam search
    python infer.py --checkpoint model.pt --text "..." --beam-size 5
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from data import Vocab, simple_tokenize, PAD_ID, BOS_ID, EOS_ID, UNK_ID
from model import GNMT


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model checkpoint and rebuild model."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Rebuild vocab
    stoi = ckpt["vocab"]
    itos = [""] * len(stoi)
    for token, idx in stoi.items():
        itos[idx] = token
    vocab = Vocab.__new__(Vocab)
    vocab.stoi = stoi
    vocab.itos = itos
    
    # Rebuild model
    args = ckpt.get("args", {})
    vocab_size = len(vocab)
    model = GNMT(
        src_vocab=vocab_size,
        tgt_vocab=vocab_size,
        embed_size=args.get("embed_size", 512),
        hidden_size=args.get("hidden_size", 512),
        num_layers=args.get("layers", 4),
        dropout=args.get("dropout", 0.2),
        pad_id=PAD_ID,
    ).to(device)
    
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    return model, vocab


def encode_text(text: str, vocab: Vocab, max_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenize and encode text to token IDs."""
    tokens = simple_tokenize(text)
    if max_len:
        tokens = tokens[:max_len]
    ids = [vocab.stoi.get(tok, UNK_ID) for tok in tokens]
    length = len(ids)
    return torch.tensor([ids], dtype=torch.long), torch.tensor([length], dtype=torch.long)


def greedy_decode(
    model: GNMT,
    src_tokens: torch.Tensor,
    src_lengths: torch.Tensor,
    vocab: Vocab,
    max_len: int = 100,
    device: torch.device = None,
) -> List[int]:
    """
    Greedy decoding: at each step, choose the token with highest probability.
    
    Returns:
        List of token IDs (excluding BOS/EOS).
    """
    if device is None:
        device = src_tokens.device
    
    batch_size = src_tokens.size(0)
    assert batch_size == 1, "Greedy decode supports batch_size=1 only"
    
    # Encode source
    with torch.no_grad():
        enc_out, enc_state = model.encoder(src_tokens, src_lengths)
        
        # Build source mask
        max_src_len = src_tokens.size(1)
        range_row = torch.arange(max_src_len, device=device).unsqueeze(0)
        src_mask = range_row >= src_lengths.unsqueeze(1)
        
        # Initialize decoder state
        h, c = enc_state
        h = [h[i] for i in range(model.decoder.num_layers)]
        c = [c[i] for i in range(model.decoder.num_layers)]
        
        # Start with BOS
        prev_token = torch.tensor([[BOS_ID]], device=device)
        generated = []
        
        for step in range(max_len):
            # Embed current token
            emb = model.decoder.embed(prev_token)
            
            # Get context from attention (use last hidden state)
            context, _ = model.decoder.attn(h[-1], enc_out, enc_out, src_mask)
            
            # First decoder layer: concat embedding + context
            layer_input = torch.cat([emb.squeeze(1), context], dim=-1)
            new_h = []
            new_c = []
            
            for idx, cell in enumerate(model.decoder.layers):
                residual = layer_input if idx >= 2 else None
                h_t, c_t = cell(layer_input, (h[idx], c[idx]))
                if residual is not None:
                    h_t = h_t + residual
                layer_input = model.decoder.dropout(h_t) if idx + 1 < model.decoder.num_layers else h_t
                new_h.append(h_t)
                new_c.append(c_t)
            
            h, c = new_h, new_c
            
            # Compute logits
            logits = model.decoder.out_proj(torch.cat([h[-1], context], dim=-1))
            probs = F.softmax(logits, dim=-1)
            
            # Greedy: choose token with highest probability
            next_token = probs.argmax(dim=-1)
            
            if next_token.item() == EOS_ID:
                break
            
            generated.append(next_token.item())
            prev_token = next_token.unsqueeze(0)
        
        return generated


def beam_search_decode(
    model: GNMT,
    src_tokens: torch.Tensor,
    src_lengths: torch.Tensor,
    vocab: Vocab,
    beam_size: int = 5,
    max_len: int = 100,
    length_penalty: float = 0.6,
    device: torch.device = None,
) -> List[int]:
    """
    Beam search decoding.
    
    Returns:
        List of token IDs (best hypothesis, excluding BOS/EOS).
    """
    if device is None:
        device = src_tokens.device
    
    batch_size = src_tokens.size(0)
    assert batch_size == 1, "Beam search supports batch_size=1 only"
    
    # Encode source (once)
    with torch.no_grad():
        enc_out, enc_state = model.encoder(src_tokens, src_lengths)
        
        max_src_len = src_tokens.size(1)
        range_row = torch.arange(max_src_len, device=device).unsqueeze(0)
        src_mask = range_row >= src_lengths.unsqueeze(1)
        
        # Initialize beams: (token_seq, log_score, hidden_states, cell_states)
        h_init, c_init = enc_state
        h_init_list = [h_init[i] for i in range(model.decoder.num_layers)]
        c_init_list = [c_init[i] for i in range(model.decoder.num_layers)]
        
        beams = [([BOS_ID], 0.0, h_init_list, c_init_list)]  # (tokens, log_score, h, c)
        finished = []
        
        for step in range(max_len):
            candidates = []
            
            for tokens, score, h, c in beams:
                if tokens[-1] == EOS_ID:
                    finished.append((tokens[1:-1], score / (len(tokens) - 1) ** length_penalty))
                    continue
                
                prev_token = torch.tensor([[tokens[-1]]], device=device)
                emb = model.decoder.embed(prev_token)
                
                # Attention: use single encoder output (shared across beams)
                context, _ = model.decoder.attn(h[-1], enc_out, enc_out, src_mask)
                
                layer_input = torch.cat([emb.squeeze(1), context], dim=-1)
                new_h = []
                new_c = []
                
                for idx, cell in enumerate(model.decoder.layers):
                    residual = layer_input if idx >= 2 else None
                    h_t, c_t = cell(layer_input, (h[idx], c[idx]))
                    if residual is not None:
                        h_t = h_t + residual
                    layer_input = model.decoder.dropout(h_t) if idx + 1 < model.decoder.num_layers else h_t
                    new_h.append(h_t)
                    new_c.append(c_t)
                
                logits = model.decoder.out_proj(torch.cat([new_h[-1], context], dim=-1))
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Get top-k candidates
                top_probs, top_indices = log_probs.topk(beam_size, dim=-1)
                
                for i in range(beam_size):
                    next_token = top_indices[0, i].item()
                    new_score = score + top_probs[0, i].item()
                    candidates.append((tokens + [next_token], new_score, new_h, new_c))
            
            if not candidates:
                break
            
            # Keep top beam_size candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_size]
        
        # Add unfinished beams
        for tokens, score, _, _ in beams:
            if tokens[-1] != EOS_ID:
                finished.append((tokens[1:], score / len(tokens) ** length_penalty))
        
        if not finished:
            return []
        
        # Return best finished hypothesis
        finished.sort(key=lambda x: x[1], reverse=True)
        return finished[0][0]


def decode_text(token_ids: List[int], vocab: Vocab) -> str:
    """Convert token IDs back to text."""
    tokens = [vocab.itos[idx] for idx in token_ids if idx not in [PAD_ID, BOS_ID, EOS_ID]]
    return " ".join(tokens)


def main():
    parser = argparse.ArgumentParser(description="GNMT inference script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--text", type=str, default=None, help="Single sentence to translate")
    parser.add_argument("--input-file", type=str, default=None, help="Input file (one sentence per line)")
    parser.add_argument("--output-file", type=str, default=None, help="Output file for translations")
    parser.add_argument("--beam-size", type=int, default=1, help="Beam size (1=greedy)")
    parser.add_argument("--max-len", type=int, default=100, help="Maximum output length")
    parser.add_argument("--length-penalty", type=float, default=0.6, help="Length penalty for beam search")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Loading checkpoint from {args.checkpoint}...")
    model, vocab = load_checkpoint(args.checkpoint, device)
    print(f"Model loaded. Vocab size: {len(vocab)}")
    
    decode_fn = beam_search_decode if args.beam_size > 1 else greedy_decode
    
    if args.text:
        # Single sentence
        print(f"\nInput: {args.text}")
        src_tokens, src_lengths = encode_text(args.text, vocab)
        src_tokens = src_tokens.to(device)
        src_lengths = src_lengths.to(device)
        
        if args.beam_size > 1:
            token_ids = decode_fn(
                model, src_tokens, src_lengths, vocab,
                beam_size=args.beam_size,
                max_len=args.max_len,
                length_penalty=args.length_penalty,
                device=device,
            )
        else:
            token_ids = decode_fn(
                model, src_tokens, src_lengths, vocab,
                max_len=args.max_len,
                device=device,
            )
        
        output = decode_text(token_ids, vocab)
        print(f"Output: {output}")
        
    elif args.input_file:
        # Batch from file
        input_path = Path(args.input_file)
        output_path = Path(args.output_file) if args.output_file else input_path.parent / f"{input_path.stem}_translated.txt"
        
        print(f"\nReading from {input_path}...")
        with open(input_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        
        print(f"Translating {len(lines)} sentences...")
        translations = []
        
        for i, line in enumerate(lines):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(lines)}")
            
            src_tokens, src_lengths = encode_text(line, vocab)
            src_tokens = src_tokens.to(device)
            src_lengths = src_lengths.to(device)
            
            if args.beam_size > 1:
                token_ids = decode_fn(
                    model, src_tokens, src_lengths, vocab,
                    beam_size=args.beam_size,
                    max_len=args.max_len,
                    length_penalty=args.length_penalty,
                    device=device,
                )
            else:
                token_ids = decode_fn(
                    model, src_tokens, src_lengths, vocab,
                    max_len=args.max_len,
                    device=device,
                )
            
            output = decode_text(token_ids, vocab)
            translations.append(output)
        
        print(f"\nWriting translations to {output_path}...")
        with open(output_path, "w", encoding="utf-8") as f:
            for trans in translations:
                f.write(trans + "\n")
        
        print(f"Done! Translated {len(translations)} sentences.")
        
    else:
        parser.error("Either --text or --input-file must be provided")


if __name__ == "__main__":
    main()

