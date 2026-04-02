"""
Universal Text Preprocessing Pipeline
======================================
Works with ANY dataset: WikiText, Sherlock Holmes, books, tweets, news, code, etc.
Just configure the PreprocessConfig and call preprocess_dataset().
"""

import re
import os
import json
import torch
import unicodedata
from pathlib import Path
from dataclasses import dataclass, field
from collections import Counter
from typing import Optional

# 1. CONFIGURATION — tweak everything from one place

@dataclass
class PreprocessConfig: # control panel
    # ── Cleaning options ──────────────────────────────────────────
    lowercase:              bool = True
    remove_html:            bool = True
    remove_urls:            bool = True
    remove_numbers:         bool = False
    remove_emails:          bool = True
    replace_numbers:        bool = True
    remove_spacial_chars:   bool = True 
    remove_extra_spaces:    bool = True
    remove_wiki_headers:    bool = True
    remove_punctuation:     bool = False
    normalize_unicode:      bool = True

    # ── Tokenization ──────────────────────────────────────────────
    tokenizer: str = "word" 

    # ── Vocabulary ────────────────────────────────────────────────
    min_freq:    int = 2                # min word freq to keep
    max_vocab:   Optional[int] = None   # cap vocab size (None = unlimited)

    # ── Sequence ──────────────────────────────────────────────────
    seq_len: int = 64
    max_tokens: Optional[int] = None  # cap total tokens loaded
   
    # ── Special tokens ────────────────────────────────────────────
    special_tokens: list = field(default_factory= lambda: [
        "<PAD>", "<UNK>", "<BOS>", "<EOS>", "<NUM>", "<URL>","<EMAIL>"
    ])


# 2. FILE LOADER — reads any plain-text file (UTF-8 or latin-1)

def load_file(path: str) -> str:
    """
    Load any plain-text file.
    Tries UTF-8 first, falls back to latin-1 so it never crashes.
    Supports: .txt, .md, .csv (raw), .json (raw), .py, etc.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            with open(path,"r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue


# 3. CLEANING STEPS — each step is independent and reusable
def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
'''
"café"
   ↓ normalize
"cafe + ` "
   ↓ encode ascii (ignore)
"cafe"
   ↓ decode
"cafe"
'''
def remove_html_tags(text:str) -> str:
    # re.sub(pattern, replacement, text)
    return re.sub(r"<[^>]+>", " ",{text})
    # "<div class='x'>Hello</div>" ====> " Hello "

def remove_wiki_headers(text: str) -> str:
    text = re.sub(r"=+\s.*?\s=+"," ", text)     # Headers
    text = re.sub(r"\[\[.*?\]\]", " ", text)    # [[wikilinks]]
    text = re.sub(r"\{\{.*?\}\}", " ", text)

def remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+" , " <URL> ", text)

def remove_emails(text: str) -> str:
    return re.sub("\S+@\S+\,\S+", " <EMAIL> ", text)

def handle_nummbers(text:str, remove: bool = False, replace: bool=False) -> str:
    if remove:
        return re.sub(r"\b\d+\b", " ", text)
    if replace:
        return re.sub("\b\d+\b", " <NUM> ", text)
    return text

def remove_special_characters(text:str) -> str:
    return re.sub(r"[^a-zA-Z0-9\s\.,!?';:\-<>]", " ", text)

def remove_punctuation(text:str) -> str:
    return re.sub(r"[^\w\s]", " ", text)

def normalize_whitespace(text:str) -> str:
    return re.sub(r"\s+"," ", text).strip()


# 4. CLEANING PIPELINE
def clean_text(text:str, cfg: PreprocessConfig) -> str:
    """    
    CRITICAL ORDERING LOGIC:
    1. Unicode & HTML: Must be first to standardize the text and remove 
       structured tags before they are broken by punctuation or case changes.
    2. URLs & Emails: Must be removed BEFORE punctuation/special characters 
       because their regex patterns rely on symbols like '.', ':', and '/'.
    3. Lowercasing: Applied after URL/HTML removal to ensure case-sensitive 
       regex patterns still work correctly.
    4. Punctuation/Special Chars: Strips remaining symbols after complex 
       entities (URLs/Emails) are already gone.
    5. Extra Spaces: Always last to clean up the 'gaps' left by the 
       previous removal steps.
    """
    if cfg.normalize_unicode:       text = normalize_unicode(text)
    if cfg.remove_html:             text = remove_html_tags(text)
    if cfg.remove_urls:             text = remove_urls(text)
    if cfg.remove_emails:           text = remove_emails(text)
    if cfg.remove_wiki_headers:     text = remove_wiki_headers(text)
    text = handle_nummbers(text, remove= cfg.remove_numbers, replace = cfg.replace_numbers)
    if cfg.lowercase : text =       text.lower()
    if cfg.remove_punctuation:      text = remove_punctuation(text)
    if cfg.remove_spacial_chars:    text = remove_special_characters(text)
    if cfg.remove_extra_spaces:     text = normalize_whitespace(text)

    return text
# 5. TOKENIZERS
def tokenizer(text:str, mode: str= "word") -> list[str]:
    if mode == "word":
        return [t for t in text.split() if t]
    if mode == "char":
        return list(text.replace(" ", "_"))
    if mode == "subword":
        return re.findall(r"\w+|[\w\s]", text)
    else:
        raise ValueError(f"unknown tokenizer mode: '{mode}'. Choose word | char | subword")

# 6. VOCABULARY BUILDER
def build_vocab(tokens: list[str], cfg: PreprocessConfig) -> dict[str,int]:

    vocab = {tok: i for i, tok in enumerate(cfg.special_tokens)}
    idx = len(vocab)
    counter = Counter(tokens)
    for word, freq in counter.most_common(cfg.max_vocab):
        if freq < cfg.min_freq:
            break
        if word not in vocab:
            vocab[word] = idx
            idx += 1
    return vocab

def save_vocab(vocab: dict, path: str):
    with open(path, "w", encoding = "utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

def load_vocab(path:str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
# 7. ENCODER

def encode(tokens: list[str], vocab: dict[str, int] )-> torch.Tensor:
    unk_id = vocab.get("<UNK>, 1")
    ids = [vocab.get(tok, unk_id) for tok in tokens]
    return torch.tensor(ids, dtype=torch.long)

def decode(ids: torch.Tensor, vocab: dict[str, int]) -> list[str]:
    id2word = {v:k for k,v in vocab.items}
    return [id2word.get(i.item(), "<UNK>") for i in ids]
# 8. SEQUENCE GENERATOR


# 9. TRAIN / VAL / TEST SPLIT


# 10. STATS — understand your data before training



# 11. ONE-CALL PIPELINE — use this in your training scripts


# 12. QUICK-START EXAMPLE


if __name__ == "__main__":

    # ── Configure ─────────────────────────────────────────────────
    
    # ── Run pipeline ──────────────────────────────────────────────
    
    # ── Use sequences in a training loop ──────────────────────────
   
    # ── Or use batched sequences ───────────────────────────────────
    
