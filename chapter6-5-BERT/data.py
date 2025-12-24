# chapter6-5-BERT/data.py

import os
import pickle
from collections import Counter
import random
from typing import Optional, Sequence, List, Tuple
from datasets import load_dataset
from tqdm import tqdm
import torch


PAD_ID = 0
UNK_ID = 100
CLS_ID = 101
SEP_ID = 102
MASK_ID = 103

class Vocab:
    def __init__(
        self,
        counter: Counter,
        min_freq: int = 1,
        max_size: Optional[int] = None,
        specials: Optional[Sequence[Tuple[str, int]]] = None,
    ):
        if specials is None:
            specials = [("<pad>", PAD_ID), ("<unk>", UNK_ID), ("<cls>", CLS_ID), ("<sep>", SEP_ID), ("<mask>", MASK_ID)]

        specials = sorted(specials, key=lambda x: x[1])
        self.itos: List[str] = [tok for tok, _ in specials]
        self.stoi = {tok: idx for tok, idx in specials}

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

        self.normal_tokens = [
            tok for tok in self.itos
            if tok not in ("<pad>", "<unk>", "<cls>", "<sep>", "<mask>")
        ]

    def __len__(self):
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def unk_id(self) -> int:
        return UNK_ID

    @property
    def cls_id(self) -> int:
        return CLS_ID

    @property
    def sep_id(self) -> int:
        return SEP_ID

    @property
    def mask_id(self) -> int:
        return MASK_ID

    def tokenize(self, text: str) -> List[str]:
        return simple_tokenize(text)

    def encode(self, tokens: Sequence[str]) -> List[int]:
        return [self.stoi.get(tok, self.unk_id) for tok in tokens]

    def decode(self, ids: Sequence[int]) -> List[str]:
        return [self.itos[i] if i < len(self.itos) else "<unk>" for i in ids]

def simple_tokenize(text: str) -> List[str]:
    """Whitespace tokenizer; the dataset is already segmented by spaces."""
    return str(text).strip().split()

COUNTER_CACHE_PATH = "token_counter.pkl"
def build_vocab(
    min_freq: int = 1,
    max_size: Optional[int] = None,
    use_cache: bool = True,
):
    if use_cache and os.path.exists(COUNTER_CACHE_PATH):
        print(f"[Counter] Loading cached counter from {COUNTER_CACHE_PATH}")
        with open(COUNTER_CACHE_PATH, "rb") as f:
            counter = pickle.load(f)
    else:
        print("[Counter] Building counter from dataset (this is slow)")
        ds = load_dataset("/dataroot/liujiang/data/datasets/bookcorpus")

        counter = Counter()
        for item in tqdm(ds["train"], desc="Counting tokens", ascii=True):
            tokens = simple_tokenize(item["text"])
            counter.update(tokens)

        with open(COUNTER_CACHE_PATH, "wb") as f:
            pickle.dump(counter, f)

        print(f"[Counter] Saved counter to {COUNTER_CACHE_PATH}")

    vocab = Vocab(counter, min_freq=min_freq, max_size=max_size)
    return vocab

class Dataset(torch.utils.data.Dataset):
    def __init__(self, vocab: Vocab, max_seq_len: int, mlm_prob: float = 0.15):
        self.max_seq_len = max_seq_len
        self.mlm_prob = mlm_prob
        self.vocab = vocab
        self.texts = load_dataset("/dataroot/liujiang/data/datasets/bookcorpus")["train"]
    
    def __len__(self):
        return len(self.texts)

    def _get_sentence_pair(self, idx: int):
        # 50% 正样本 / 50% 负样本
        if random.random() < 0.5:
            # 正样本：直接用 idx 和 idx+1
            textA = self.texts[idx]["text"]
            # 避免 idx+1 越界
            next_idx = min(idx + 1, len(self.texts) - 1)
            textB = self.texts[next_idx]["text"]
            is_next = 1
        else:
            # 负样本：B 来自随机文章
            textA = self.texts[idx]["text"]
            rand_idx = random.randint(0, len(self.texts) - 1)
            while rand_idx == idx:
                rand_idx = random.randint(0, len(self.texts) - 1)
            textB = self.texts[rand_idx]["text"]
            is_next = 0

        return textA, textB, is_next

    def _truncate_pair(self, tokens_a, tokens_b, max_len):
        while len(tokens_a) + len(tokens_b) > max_len:
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def _token_masking(self, tokens: List[str]):
        """
        对 tokens 进行 MLM masking（in-place 修改 tokens）
        返回:
            mlm_labels: List[int]，未被 mask 的位置为 -100
        """
        # nn.CrossEntropyLoss 计算loss时忽略 -100 的位置
        # 这样未被 mask 的位置不会对 loss 有贡献
        mlm_labels = [-100] * len(tokens)

        for i, tok in enumerate(tokens):
            # 跳过特殊 token
            if tok in ("<cls>", "<sep>", "<pad>"):
                continue

            if random.random() < self.mlm_prob:
                # 记录监督信号（原 token id）
                mlm_labels[i] = self.vocab.stoi.get(tok, self.vocab.unk_id)

                prob = random.random()
                if prob < 0.8:
                    # 80% -> [mask]
                    tokens[i] = "<mask>"
                elif prob < 0.9:
                    # 10% -> 随机词（不能是特殊 token）
                    tokens[i] = random.choice(self.vocab.normal_tokens)
                # 10% -> 保持不变

        return mlm_labels
    
    def collate_fn(self, batch):
        input_ids, token_type_ids, mlm_labels, is_next = zip(*batch)

        def pad(seq, pad_val):
            return seq + [pad_val] * max(0, self.max_seq_len - len(seq))

        attention_mask = [
            pad([1] * len(ids), 0)
            for ids in input_ids
        ]

        batch = {
            "input_ids": torch.tensor(
                [pad(ids, self.vocab.pad_id) for ids in input_ids],
                dtype=torch.long,
            ),
            "token_type_ids": torch.tensor(
                [pad(tt, 0) for tt in token_type_ids],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                [pad(ml, -100) for ml in mlm_labels],
                dtype=torch.long,
            ),
            "next_sentence_label": torch.tensor(is_next, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        }

        return batch

    def __getitem__(self, idx: int):
        textA, textB, is_next = self._get_sentence_pair(idx)

        tokens_a = self.vocab.tokenize(textA)
        tokens_b = self.vocab.tokenize(textB)

        # 为 [CLS] + 2*[SEP] 预留 3 个位置
        max_pair_len = self.max_seq_len - 3

        # 联合裁剪
        self._truncate_pair(tokens_a, tokens_b, max_pair_len)
        tokens = (
            ["<cls>"] +
            tokens_a +
            ["<sep>"] +
            tokens_b +
            ["<sep>"]
        )

        mlm_labels = self._token_masking(tokens)

        token_type_ids = (
            [0] * (len(tokens_a) + 2) +
            [1] * (len(tokens_b) + 1)
        )
        assert len(token_type_ids) == len(tokens)

        input_ids = self.vocab.encode(tokens)

        return input_ids, token_type_ids, mlm_labels, is_next

def build_dataloader(
        min_freq: int = 10,
        max_size: Optional[int] = None,
        max_seq_len: int = 128,
        mlm_prob: float = 0.15,
        batch_size: int = 32,
        num_workers: int = 0,
):
    vocab = build_vocab(min_freq=min_freq, max_size=max_size)
    dataset = Dataset(vocab=vocab, max_seq_len=max_seq_len, mlm_prob=mlm_prob)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        drop_last=True,
        num_workers=num_workers,
    )
    return dataloader


if __name__ == "__main__":
    # vocab = build_vocab(min_freq=20)
    # dataset = Dataset(vocab=vocab, max_seq_len=128, mlm_prob=0.15)
    # input_ids, token_type_ids, mlm_labels, is_next = dataset[0]
    # print(len(input_ids), len(token_type_ids), len(mlm_labels))
    # print("Input IDs:", input_ids)
    # print("Token Type IDs:", token_type_ids)
    # print("MLM Labels:", mlm_labels)
    # print("Is Next Label:", is_next)

    dataloader = build_dataloader(
        min_freq=20,
        max_size=30000,
        max_seq_len=128,
        mlm_prob=0.15,
        batch_size=16,
        num_workers=0,
    )
    for batch in dataloader:
        print("Input IDs:", batch["input_ids"].shape)
        print("Token Type IDs:", batch["token_type_ids"].shape)
        print("Attention Mask:", batch["attention_mask"].shape)
        print("MLM Labels:", batch["labels"].shape)
        print("Is Next Labels:", batch["next_sentence_label"].shape)
        break