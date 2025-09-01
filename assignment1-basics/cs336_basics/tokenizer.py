import os
import json
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

from cs336_basics.train_bpe import _tokenize, word_to_byte_tuple


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        if special_tokens is not None:
            for sp in special_tokens:
                b_sp = sp.encode('utf-8') 
                if b_sp not in vocab.values():
                    print(f"Warning: Special tokens {special_tokens} not found in vocab. Adding with id {len(vocab)}.")
                    vocab[len(vocab)] = b_sp   
        self.vocab_to_ids = {v: k for k, v in vocab.items()}
        self.ids_to_vocab = vocab
        
        rev = defaultdict(list)
        for k, v in vocab.items():
            rev[v].append(k)

        dupes = {val: ids for val, ids in rev.items() if len(ids) > 1}
        for token, ids in dupes.items():
            print(f"Token {token!r} has multiple ids: {ids}")

        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "r") as vocab_file:
            vocab_str = json.load(vocab_file)
            vocab = {int(v): k.encode('utf-8') for k, v in vocab_str.items()}
            
        with open(merges_filepath, "r") as merges_file:
            merges_str = merges_file.read().splitlines()
            merges = [(token1.encode('utf-8'), token2.encode('utf-8')) for token1, token2 in merges_str]
        
        return cls(vocab, merges, special_tokens)
    
    def _encode_to_ids(self, token):        
        if not isinstance(token, list):
            token = list(token)
            
        if len(token) > 1:
            for m1, m2 in self.merges:
                i = 0
                while i < len(token) - 1:
                    if token[i] == m1 and token[i + 1] == m2:
                        token[i:i + 2] = [m1 + m2]
                    else:
                        i += 1

        ids = [self.vocab_to_ids.get(b) for b in token]
        if any(x is None for x in ids):
            missing = [token[k] for k, x in enumerate(ids) if x is None]
            raise KeyError(f"Symbols not in vocab: {missing}")
        return ids
        
    
    def encode(self, text, max_workers=1):
        if not text:
            return []
        pre_tokens = _tokenize(text, list(self.special_tokens), return_raw=True)
        tokens = []
        for t in pre_tokens:
            if t in self.special_tokens:
                tokens.append(self.vocab_to_ids.get(t.encode('utf-8')))
            else:
                tokens.append(word_to_byte_tuple(t))
        if max_workers is None:
            max_workers = os.cpu_count() or 1
        max_workers = min(max_workers, len(tokens))
        if max_workers <= 1:
            out = []
            for tok in tokens:
                if isinstance(tok, int): # special token
                    out.append(tok)
                else:
                    out.extend(self._encode_to_ids(tok))
            return out
        
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            parts = list(ex.map(self._encode_to_ids, tokens))
        out = []
        for p in parts:
            out.extend(p)
        return out
        
    
    def encode_iterable(self, iterable):
        for text in iterable:
            yield from self.encode(text)
            
    def decode(self, ids):
        if not ids:
            return ""
        if not isinstance(ids, list):
            ids = [ids]
        b_decoded = b"".join([self.ids_to_vocab.get(id) for id in ids])
        return b_decoded.decode('utf-8', errors='replace')