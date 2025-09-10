import os
import json
from datasets import load_dataset
from tqdm import tqdm

# 参数
DATASET_PATH = "/datasets/preset/cerebras"
OUTPUT_DIR = "/volume/bhzhao-data/dataset/cerebras"
NUM_SHARDS = 32
NUM_LINES_TOTAL = 1000000

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 打开多个 shard 输出文件
writers = [open(os.path.join(OUTPUT_DIR, f"shard_{i}.jsonl"), "w") for i in range(NUM_SHARDS)]

# 流式读取
ds = load_dataset(DATASET_PATH, split="train", streaming=True)
for i, example in tqdm(enumerate(ds), total=NUM_LINES_TOTAL):
    shard_id = i % NUM_SHARDS
    json.dump(example, writers[shard_id])
    writers[shard_id].write("\n")
    if i + 1 >= NUM_LINES_TOTAL:
        break

for w in writers:
    w.close()

print(f"✅ 完成切分，写入 {NUM_SHARDS} 个 shard 文件，每个大约 {NUM_LINES_TOTAL // NUM_SHARDS} 条")
