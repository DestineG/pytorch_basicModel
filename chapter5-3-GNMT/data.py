"""
Dataset utilities for the bilingual (zh-en) translation corpus.
The CSV is expected to have two columns: source (zh) and target (en).
Default column indices are 0 (source) and 1 (target) to match the sample file.
"""

from collections import Counter
from typing import Callable, List, Optional, Sequence, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


def simple_tokenize(text: str) -> List[str]:
    """Whitespace tokenizer; the dataset is already segmented by spaces."""
    return str(text).strip().split()


class Vocab:
    def __init__(
        self,
        counter: Counter,
        min_freq: int = 1,
        max_size: Optional[int] = None,
        specials: Optional[Sequence[Tuple[str, int]]] = None,
    ):
        # specials lets us control PAD/BOS/EOS/UNK ids explicitly
        if specials is None:
            specials = [("<pad>", PAD_ID), ("<bos>", BOS_ID), ("<eos>", EOS_ID), ("<unk>", UNK_ID)]
        # sort specials by id to keep stable ordering
        specials = sorted(specials, key=lambda x: x[1])
        self.itos: List[str] = [tok for tok, _ in specials]
        self.stoi = {tok: idx for tok, idx in specials}

        # build from counter
        tokens_and_freq = counter.most_common()
        if max_size is not None:
            tokens_and_freq = tokens_and_freq[: max(0, max_size - len(self.itos))]

        for token, freq in tokens_and_freq:
            # 若不满足最低频次
            if freq < min_freq:
                continue
            # 若已在字典中
            if token in self.stoi:
                continue
            # 更新字典
            self.stoi[token] = len(self.itos)
            # 更新列表
            self.itos.append(token)

    def __len__(self):
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    @property
    def unk_id(self) -> int:
        return UNK_ID

    def encode(self, tokens: Sequence[str]) -> List[int]:
        return [self.stoi.get(tok, self.unk_id) for tok in tokens]

    def decode(self, ids: Sequence[int]) -> List[str]:
        return [self.itos[i] if i < len(self.itos) else "<unk>" for i in ids]


def build_vocab(
    texts: Sequence[Sequence[str]],
    min_freq: int = 1,
    max_size: Optional[int] = None,
) -> Vocab:
    counter = Counter()
    for toks in texts:
        counter.update(toks)
    return Vocab(counter, min_freq=min_freq, max_size=max_size)


class TranslationDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        src_col: Union[int, str] = 0,
        tgt_col: Union[int, str] = 1,
        vocab: Optional[Vocab] = None,
        tokenizer: Callable[[str], List[str]] = simple_tokenize,
        min_freq: int = 1,
        max_size: Optional[int] = None,
    ):
        super().__init__()

        # 卡在这里了 减少数据量
        print("Reading data...")
        self.df = pd.read_csv(csv_path, nrows=20000)
        self.src_col = src_col
        self.tgt_col = tgt_col
        self.tokenizer = tokenizer

        # Support both integer (positional) and string (column name) indexing
        # If column is integer, use iloc for positional access (works with or without header)
        # If column is string, use column name access
        print("Splitting data...")
        if isinstance(src_col, int):
            src_series = self.df.iloc[:, src_col]
        else:
            src_series = self.df[src_col]
        
        if isinstance(tgt_col, int):
            tgt_series = self.df.iloc[:, tgt_col]
        else:
            tgt_series = self.df[tgt_col]

        print("Tokenizing dataset...")
        src_texts = [tokenizer(str(t)) for t in src_series.tolist()]
        tgt_texts = [tokenizer(str(t)) for t in tgt_series.tolist()]

        print("Building vocabulary...")
        if vocab is None:
            # build a shared vocab over both sides (simple but sufficient)
            self.vocab = build_vocab(src_texts + tgt_texts, min_freq=min_freq, max_size=max_size)
        else:
            self.vocab = vocab

        # 将语句离散化，添加起始符和结束符
        # List[List[str]] -> List[List[int]]
        print("Numericalizing dataset...")
        self.src = [self._numericalize(toks) for toks in src_texts]
        self.tgt = [self._numericalize(toks) for toks in tgt_texts]

    def _numericalize(self, tokens: Sequence[str]) -> List[int]:
        # Add BOS/EOS for seq2seq training
        return [self.vocab.bos_id] + self.vocab.encode(tokens) + [self.vocab.eos_id]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        return self.src[idx], self.tgt[idx]

    def collate_fn(self, batch: List[Tuple[List[int], List[int]]]):
        # 源语句列表,目标语句列表
        src_seqs, tgt_seqs = zip(*batch)
        src_lengths = torch.tensor([len(s) for s in src_seqs], dtype=torch.long)
        tgt_lengths = torch.tensor([len(t) for t in tgt_seqs], dtype=torch.long)

        # 源语句和目标语句的最大长度
        max_src = max(src_lengths).item()
        max_tgt = max(tgt_lengths).item()

        # 在句子尾部标识符后填充，使其长度一致
        def pad(seqs, max_len):
            # 初始化输出张量，全部填充为pad_id
            out = torch.full((len(seqs), max_len), self.vocab.pad_id, dtype=torch.long)
            for i, seq in enumerate(seqs):
                # 将语句填充到对应位置
                out[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            return out

        src_tensor = pad(src_seqs, max_src)
        tgt_tensor = pad(tgt_seqs, max_tgt)
        return src_tensor, src_lengths, tgt_tensor


def create_dataloader(
    csv_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    src_col: Union[int, str] = 0,
    tgt_col: Union[int, str] = 1,
    vocab: Optional[Vocab] = None,
    tokenizer: Callable[[str], List[str]] = simple_tokenize,
    min_freq: int = 1,
    max_size: Optional[int] = None,
) -> Tuple[DataLoader, TranslationDataset]:
    print("Building dataset...")
    dataset = TranslationDataset(
        csv_path=csv_path,
        src_col=src_col,
        tgt_col=tgt_col,
        vocab=vocab,
        tokenizer=tokenizer,
        min_freq=min_freq,
        max_size=max_size,
    )
    print("Building loader...")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
    )
    return loader, dataset
