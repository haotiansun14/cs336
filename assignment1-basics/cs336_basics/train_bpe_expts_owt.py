import os
import json
from cs336_basics.train_bpe import train_bpe
from tests.common import gpt2_bytes_to_unicode

TRAIN_PATH = "data/owt_train.txt"
VALID_PATH = "data/owt_valid.txt"

VOCAB_SIZE = 32_000

SPECIALS = ["<|endoftext|>"]

CORPUS_NAME = "owt"
BYTE_TO_UNICODE = gpt2_bytes_to_unicode()  # {0..255 -> printable Unicode}

# [profile] pretokenize: 13.639s, merges: 39.243s, total: 52.882s

def bytes_to_token_str(b: bytes) -> str:
    """Try strict UTF-8; if it fails, map each raw byte via GPT-2's bijection."""
    try:
        return b.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return "".join(BYTE_TO_UNICODE[x] for x in b)

if __name__ == "__main__":
    vocab_dict, merges = train_bpe(TRAIN_PATH, VOCAB_SIZE, SPECIALS, num_processes=-1)
    print(f"Vocabulary size: {len(vocab_dict)}")
    print(f"Merges: {merges[:20]}")
    
    # Save the learned vocab and merges for future use
    os.makedirs(f"tokenizers/{CORPUS_NAME}", exist_ok=True)
    
    vocab_str = {}
    for idx, b_vocab in vocab_dict.items():
        token_str = bytes_to_token_str(b_vocab)
        vocab_str[token_str] = idx
    
    # get the longest token in the vocabulary
    longest_token = max(vocab_str.keys(), key=len)
    print(f"Longest token: {longest_token}")
    
    with open(f"tokenizers/{CORPUS_NAME}/vocab.json", "w") as f:
        print(f"Saving vocabulary to tokenizers/{CORPUS_NAME}/vocab.json")
        json.dump(vocab_str, f, ensure_ascii=False, indent=2)

    # breakpoint()
    with open(f"tokenizers/{CORPUS_NAME}/merges.txt", "w") as f:
        print(f"Saving merges to tokenizers/{CORPUS_NAME}/merges.txt")
        for merge_token_1, merge_token_2 in merges:
            s1 = bytes_to_token_str(merge_token_1)
            s2 = bytes_to_token_str(merge_token_2)
            f.write(f"[{s1}]+[{s2}]\n")
    