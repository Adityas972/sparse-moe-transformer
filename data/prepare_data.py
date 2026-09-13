"""Downloads TinyShakespeare and tokenizes it at the character level
(vocab_size ~65). Went with char-level over BPE mainly for simplicity - no
merge table to train/load, no extra dependency, and this project is about
the attention/MoE internals, not the tokenizer. Costs ~4x longer sequences
for the same text vs. a subword tokenizer, which matters less at this
scale than the reduced moving parts do. Swapping in BPE later would only
touch this file plus meta.pkl's vocab_size in train.py.
"""
import os
import pickle
import urllib.request

import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(DATA_DIR, "input.txt")
TRAIN_BIN = os.path.join(DATA_DIR, "train.bin")
VAL_BIN = os.path.join(DATA_DIR, "val.bin")
META_PATH = os.path.join(DATA_DIR, "meta.pkl")

TINYSHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def download_if_missing():
    if os.path.exists(INPUT_PATH):
        return
    print(f"Downloading TinyShakespeare to {INPUT_PATH} ...")
    try:
        urllib.request.urlretrieve(TINYSHAKESPEARE_URL, INPUT_PATH)
    except Exception as e:
        raise RuntimeError(
            f"Could not download TinyShakespeare ({e}). If you're offline, manually "
            f"download it from {TINYSHAKESPEARE_URL} and save it to {INPUT_PATH}."
        )


def prepare():
    download_if_missing()
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Loaded {len(text):,} characters.")

    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    print(f"Vocab size: {vocab_size} unique characters.")

    def encode(s: str) -> list[int]:
        return [stoi[c] for c in s]

    n = len(text)
    train_text = text[: int(n * 0.9)]
    val_text = text[int(n * 0.9) :]

    train_ids = np.array(encode(train_text), dtype=np.uint16)
    val_ids = np.array(encode(val_text), dtype=np.uint16)
    print(f"Train: {len(train_ids):,} tokens, Val: {len(val_ids):,} tokens.")

    train_ids.tofile(TRAIN_BIN)
    val_ids.tofile(VAL_BIN)

    with open(META_PATH, "wb") as f:
        pickle.dump({"vocab_size": vocab_size, "stoi": stoi, "itos": itos}, f)

    print(f"Wrote {TRAIN_BIN}, {VAL_BIN}, {META_PATH}")


if __name__ == "__main__":
    prepare()
