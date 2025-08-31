import os
from collections import Counter, defaultdict
from multiprocessing import Pool
import regex as re
from tqdm.auto import tqdm

from cs336_basics.pretokenization_example import find_chunk_boundaries  

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_PAT = re.compile(PAT, re.V1)

EOT_TOKEN = b"<|endoftext|>"

def word_to_byte_tuple(word: str):
    if not word:
        return (b"",)  
    return tuple(bytes([b]) for b in word.encode("utf-8"))

def get_freq_table(token_list):
    """Return {byte_tuple_token: freq}."""
    freq = Counter(token_list)
    return {word_to_byte_tuple(tok): c for tok, c in freq.items()}

def get_bp_freq_table(token_list):
    """
    Returns:
      bp_freq_table: Counter over adjacent byte-pairs
      token_index: { (b_i, b_{i+1}) -> set of (token_bytes_tuple, i) occurrences }
    """
    freq_table_bytes = get_freq_table(token_list)
    bp_freq_table = Counter()
    token_index = defaultdict(set)
    for btok, f in freq_table_bytes.items():
        for i in range(len(btok) - 1):
            pair = (btok[i], btok[i + 1])
            bp_freq_table[pair] += f
            token_index[pair].add((btok, i))
    return bp_freq_table, token_index

def _tokenize(text, special_tokens):
    if not special_tokens:
        return [m.group(0) for m in _PAT.finditer(text)]
    specials_pattern = re.compile("|".join(special_tokens))
    out = []
    parts = specials_pattern.split(text)
    for part in parts:
        if not part:
            continue
        if part not in special_tokens:
            out.extend(m.group(0) for m in _PAT.finditer(part))

    return out

def _process_span(args):
    """Worker: read a file span, decode, tokenize, build stats."""
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    toks = _tokenize(chunk, special_tokens)
    # toks = chunk.split() # test only
    chunk_freq = get_freq_table(toks)
    chunk_bp, chunk_index = get_bp_freq_table(toks)
    return chunk_freq, chunk_bp, chunk_index

def get_most_freq_pair(bp_freq_table):
    return max(bp_freq_table.items(), key=lambda kv: (kv[1], kv[0]))


def one_merging_step(bp_freq_table, token_freq_table, token_index, *, direction="ltr"):
    if not bp_freq_table:
        return None, 0

    (x, y), top_freq = get_most_freq_pair(bp_freq_table)
    merged = x + y

    occurrences = list(token_index.pop((x, y), set()))
    tok_pos = defaultdict(list)
    for tok, i in occurrences:
        tok_pos[tok].append(i)

    def bump(bp, freq_delta):
        if bp is None:
            return
        new = bp_freq_table.get(bp, 0) + freq_delta
        if new > 0:
            bp_freq_table[bp] = new
        else:
            bp_freq_table.pop(bp, None)

    def remove_token_from_index(tok_to_remove):
        if len(tok_to_remove) < 2:
            return
        for i in range(len(tok_to_remove) - 1):
            pair = (tok_to_remove[i], tok_to_remove[i + 1])
            s = token_index.get(pair)
            if s is None:
                continue
            s.discard((tok_to_remove, i))
            if not s:
                token_index.pop(pair, None)

    def add_token_to_index(tok_to_add):
        if len(tok_to_add) < 2:
            return
        for i in range(len(tok_to_add) - 1):
            pair = (tok_to_add[i], tok_to_add[i + 1])
            token_index.setdefault(pair, set()).add((tok_to_add, i))

    for tok, pos in tok_pos.items():
        if tok not in token_freq_table:
            continue
        freq = token_freq_table[tok]

        cur = list(tok)

        if direction == "rtl":
            # Right-to-left: indices don't shift for earlier positions
            pos.sort(reverse=True)
            last_merged_left_idx = float("inf")  
            for i in pos:
                if i + 1 >= last_merged_left_idx:
                    continue
                if i + 1 >= len(cur) or cur[i] != x or cur[i + 1] != y:
                    continue

                cur[i:i + 2] = [merged]
                last_merged_left_idx = i  

        else:  # direction == "ltr"
            pos.sort() 
            last_orig_i = -2  
            offset = 0      
            for i in pos:
                if i <= last_orig_i + 1:
                    continue
                j = i - offset
                if j + 1 >= len(cur) or cur[j] != x or cur[j + 1] != y:
                    continue

                cur[j:j + 2] = [merged]
                last_orig_i = i
                offset += 1  

        new_tok = tuple(cur)
        if new_tok == tok:
            continue

        # Update bp_freq_table: remove old pairs, add new pairs (weighted by token frequency)
        old_pairs = Counter((tok[i], tok[i + 1]) for i in range(len(tok) - 1))
        new_pairs = Counter((new_tok[i], new_tok[i + 1]) for i in range(len(new_tok) - 1))
        for p, c in old_pairs.items():
            bump(p, -freq * c)
        for p, c in new_pairs.items():
            bump(p, +freq * c)

        # Update indexes and token frequencies
        remove_token_from_index(tok)
        add_token_to_index(new_tok)
        token_freq_table.pop(tok)
        token_freq_table[new_tok] = token_freq_table.get(new_tok, 0) + freq

    # Remove the merged pair from bp_freq_table as a safeguard
    bp_freq_table.pop((x, y), None)
    return (x, y), top_freq



def train_bpe(input_path, vocab_size, special_tokens, num_processes=1):
    if num_processes in (-1, 0, None):
        num_processes = os.cpu_count() or 1
    if not isinstance(special_tokens, list):
        special_tokens = [special_tokens]
    merges = []
    freq_table = Counter()         # {token_bytes_tuple: count}
    bp_freq_table = Counter()      # {(b_i, b_{i+1}): count}
    token_index = defaultdict(set) # {(b_i, b_{i+1}): set[(token_bytes_tuple, i)]}

    # Compute chunk boundaries
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, EOT_TOKEN)
    spans = list(zip([input_path] * (len(boundaries) - 1), boundaries[:-1], boundaries[1:], [special_tokens] * (len(boundaries) - 1)))
    # Parallel pre-tokenization
    if num_processes > 1 and len(spans) > 1:
        with Pool(processes=num_processes) as pool:
            for cf, cbp, cidx in tqdm(pool.imap_unordered(_process_span, spans),
                                      total=len(spans), desc="Pretokenize"):
                freq_table.update(cf)
                bp_freq_table.update(cbp)
                for k, s in cidx.items():
                    token_index[k] |= s
    else:
        for job in tqdm(spans, desc="Pretokenize"):
            cf, cbp, cidx = _process_span(job)
            freq_table.update(cf)
            bp_freq_table.update(cbp)
            for k, s in cidx.items():
                token_index[k] |= s
    # Merge loop
    vocab_bytes = [tok.encode("utf-8") for tok in special_tokens] + [bytes([i]) for i in range(256)]
    steps = max(0, vocab_size - len(vocab_bytes))
    for _ in tqdm(range(steps), desc="Merging"):
        merged_pair, _ = one_merging_step(bp_freq_table, freq_table, token_index)
        merges.append(merged_pair)
    vocab_bytes.extend([b"".join(m) for m in merges])
    vocab_dict = {tid: btok for tid, btok in enumerate(vocab_bytes)}

    return vocab_dict, merges

if __name__ == "__main__":
    vocab_size = 14
    input_path = "data/test.txt" #"assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
    special_tokens = ["", "!!!!!"]

    vocab, merges = train_bpe(input_path, vocab_size, special_tokens, num_processes=-1)
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Merges: {merges[:15]}")