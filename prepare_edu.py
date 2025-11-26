import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import struct

# Configuration
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
SUBSET = "sample-10BT"
OUTPUT_DIR = "data/finewebEDU"
SHARD_SIZE = 100_000_000 # 100M tokens per shard

def write_datafile(filename, tokens_np):
    """Writes tokens to a binary file with the specific header format required by the loader."""
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520 # Magic
    header[1] = 1        # Version
    header[2] = len(tokens_np) # Token count
    
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens_np.tobytes())

def process_shard(args):
    shard_idx, shard_data = args
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens['<|endoftext|>']
    
    all_tokens = []
    for text in shard_data:
        tokens = enc.encode(text, allowed_special={'<|endoftext|>'})
        all_tokens.extend(tokens)
        all_tokens.append(eot)
    
    tokens_np = np.array(all_tokens, dtype=np.uint16)
    filename = os.path.join(OUTPUT_DIR, f"fineweb_edu_train_{shard_idx:06d}.bin")
    write_datafile(filename, tokens_np)
    return len(tokens_np)

def prepare():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Dataset (Streaming is safer for disk space, but we need a good chunk)
    print(f"Downloading {DATASET_NAME}...")
    # We take the first 1GB of text roughly to create our training set
    ds = load_dataset(DATASET_NAME, name=SUBSET, split="train", streaming=True)
    
    # 2. Iterate and buffer
    buffer = []
    shard_idx = 0
    token_count = 0
    
    print("Tokenizing and sharding...")
    # Use a generator to batch data for multiprocessing
    iter_ds = iter(ds)
    
    # Create validation set first (small)
    val_buffer = []
    for _ in range(1000): # ~1M tokens for val
        val_buffer.append(next(iter_ds)['text'])
    
    # Write Val
    print("Writing Validation Shard...")
    process_shard((-1, val_buffer))
    os.rename(
        os.path.join(OUTPUT_DIR, f"fineweb_edu_train_{-1:06d}.bin"),
        os.path.join(OUTPUT_DIR, "fineweb_edu_val_000000.bin")
    )

    # Training Loop
    pool = mp.Pool(mp.cpu_count())
    batch_size = 10000 # Documents per shard batch
    
    while True:
        batch = []
        try:
            for _ in range(batch_size):
                batch.append(next(iter_ds)['text'])
        except StopIteration:
            if batch:
                pool.apply(process_shard, args=((shard_idx, batch),))
            break
            
        if not batch: break
        
        # Simple sequential processing for safety/simplicity in this script
        # (Parallelizing the tokenizer inside the process_shard is better if you have massive CPUs)
        ntok = process_shard((shard_idx, batch))
        token_count += ntok
        print(f"Shard {shard_idx} written: {ntok/1e6:.2f}M tokens. Total: {token_count/1e6:.2f}M")
        shard_idx += 1
        
        if token_count > 2_000_000_000: # Stop after 2B tokens (plenty for speedrun)
            break

    print(f"Done. Data saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    prepare()