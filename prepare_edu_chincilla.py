import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import glob

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
SUBSET = "sample-10BT"
OUTPUT_DIR = "data/finewebEDU"
TARGET_TOKENS = 3_000_000_000 # 3 Billion tokens (2.5B target + buffer)

def write_datafile(filename, tokens_np):
    """Writes tokens to a binary file with the specific header."""
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
        # Encode with special tokens allowed
        tokens = enc.encode(text, allowed_special={'<|endoftext|>'})
        all_tokens.extend(tokens)
        all_tokens.append(eot)
    
    tokens_np = np.array(all_tokens, dtype=np.uint16)
    filename = os.path.join(OUTPUT_DIR, f"fineweb_edu_train_{shard_idx:06d}.bin")
    write_datafile(filename, tokens_np)
    return len(tokens_np)

def prepare():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Check for existing progress
    existing_shards = sorted(glob.glob(os.path.join(OUTPUT_DIR, "fineweb_edu_train_*.bin")))
    start_shard_idx = 0
    current_tokens = 0
    
    if existing_shards:
        print(f"Found {len(existing_shards)} existing shards. Resuming...")
        start_shard_idx = len(existing_shards)
        # Roughly estimate tokens (filesize / 2 bytes per token)
        current_tokens = sum(os.path.getsize(f) for f in existing_shards) // 2
        print(f"Estimated existing tokens: {current_tokens/1e9:.2f}B")

    if current_tokens >= TARGET_TOKENS:
        print("Data target already met. Exiting.")
        return

    # 2. Load Dataset
    print(f"Downloading {DATASET_NAME} (Stream)...")
    ds = load_dataset(DATASET_NAME, name=SUBSET, split="train", streaming=True)
    iter_ds = iter(ds)

    # 3. Validation Set (Only if missing)
    val_path = os.path.join(OUTPUT_DIR, "fineweb_edu_val_000000.bin")
    if not os.path.exists(val_path):
        print("Generating Validation Shard...")
        val_buffer = []
        for _ in range(2000): # ~2-3M tokens
            try:
                val_buffer.append(next(iter_ds)['text'])
            except StopIteration:
                break
        
        # We use a temporary ID for processing, then rename
        process_shard((-1, val_buffer))
        os.rename(os.path.join(OUTPUT_DIR, f"fineweb_edu_train_{-1:06d}.bin"), val_path)

    # 4. Training Loop
    pool = mp.Pool(max(1, mp.cpu_count() - 1)) # Leave 1 core for system
    batch_size = 10000 # Docs per shard
    shard_idx = start_shard_idx
    
    # Skip data we technically "processed" if resuming (imperfect but fast forward)
    # Ideally, dataset streaming resumption is harder, so we just pull fresh data
    # assuming the stream is shuffled or sufficiently random.
    
    pbar = tqdm(total=TARGET_TOKENS, initial=current_tokens, unit="tok", desc="Processing")
    
    while current_tokens < TARGET_TOKENS:
        batch = []
        try:
            for _ in range(batch_size):
                batch.append(next(iter_ds)['text'])
        except StopIteration:
            break
            
        if not batch: break
        
        # Process
        ntok = process_shard((shard_idx, batch))
        current_tokens += ntok
        shard_idx += 1
        pbar.update(ntok)

    pbar.close()
    pool.close()
    print(f"Done. Total Shards: {shard_idx}. Data saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    prepare()