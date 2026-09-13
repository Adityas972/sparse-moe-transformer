"""
Downloads TinyShakespeare and prepares it as a character-level dataset.

Tokenizer choice -- character-level vs. BPE:
  This project uses a character-level tokenizer (vocab_size ~65: the letters,
  punctuation, and whitespace that actually appear in the text). The
  tradeoff versus a BPE/subword tokenizer:

    - Character-level: trivial to implement (no tokenizer training, no
      external dependency), tiny vocab (cheap embedding/output matrices,
      which matters when d_model=256), and every possible input is
      representable. The cost is that sequences are ~4x longer for the same
      text (average English word/token ratio), so the model has to spend
      capacity learning spelling and basic word structure before it can get
      to anything higher-level, and needs a larger context_length to "see"
      the same amount of text a subword tokenizer would fit in fewer tokens.
    - BPE/subword: much shorter sequences for the same text (GPT-2's BPE
      gets ~4 chars/token), so the model reaches word- and phrase-level
      patterns faster per training step, and context_length covers more
      actual content. The cost is a whole extra pipeline (training or
      loading a merge table, handling of unknown byte sequences) that adds
      moving parts to a project whose actual point is the MoE layer, not
      tokenization.

  For a small, fast-iterating educational run, character-level keeps the
  moving parts down to exactly the ones this project is about (attention,
  RoPE, MoE routing) -- so that's what's implemented here. Swapping in a BPE
  tokenizer later would only require changing this file and reading
  meta.pkl's vocab_size in train.py; nothing else in the model depends on
  the tokenization scheme.
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
