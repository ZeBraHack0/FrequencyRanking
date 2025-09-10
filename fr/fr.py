from transformers import AutoTokenizer
from collections import Counter
from multiprocessing import Pool
from tqdm import tqdm
import torch
import argparse
import os
import json

MODEL_PATH = None
NUM_SHARDS = 32
SHARD_DIR = "/volume/bhzhao-data/dataset/cerebras"

def process_shard(shard_id):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
    counter = Counter()
    num_tokens = 0
    shard_path = f"{SHARD_DIR}/shard_{shard_id}.jsonl"

    with open(shard_path, "r") as f:
        for line in f:
            example = json.loads(line)
            tokens = tokenizer.encode(example["text"], truncation=True)
            counter.update(tokens)
            num_tokens += len(tokens)
    return counter, num_tokens

def main(args):
	with Pool(NUM_SHARDS) as pool:
		results = pool.map(process_shard, range(NUM_SHARDS))

    # 合并计数器
	total_counter = Counter()
	num_tokens = 0
	for counter, num in results:
		total_counter.update(counter)
		num_tokens += num

    # 排序输出
	sort_by_freq = sorted(total_counter.items(), key=lambda x: x[1], reverse=True)
	ids, frequencies = zip(*sort_by_freq)
	ids = list(ids)

	print(f"processed {NUM_LINES} items")
	print(f"processed {num_tokens} tokens")

	if not os.path.exists(f'fr_index/{args.model_name}'):
			os.makedirs(f'fr_index/{args.model_name}')
   
	tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
			
	for r in args.vocab_size:
		eos_id = tokenizer.encode(tokenizer.special_tokens_map['eos_token'])
		if eos_id not in ids[:r]:
			not_in_ids = len(set(eos_id) - set(ids[:r]))
			freq_ids = ids[:r - not_in_ids] + eos_id
		else:
			freq_ids = ids[:r]
   
		cdf = 0
		for k in range(r):
			cdf += frequencies[k]
			if (k + 1) % 256 == 0 or (k + 1) == r:
				percent = cdf / num_tokens * 100
				# print(f'cdf of top {k + 1} tokens: {cdf} ({percent:.2f}%)')
				print(f'{percent:.2f}%')

		print(f'save freq_{r}.pt, size:', len(freq_ids))
		with open(f'fr_index/{args.model_name}/freq_{r}.pt', 'wb') as f:
			torch.save(freq_ids, f)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument(
		'--model_name', 
		type=str, 
		default='llama3-8b-instruct',
		help='The name of the model.'
	)
	parser.add_argument(
		'--model_path', 
		type=str, 
		default='meta-llama/Llama-3-8B-Instruct',
		help='The path to the model.'
	)
	parser.add_argument(
		'--num_lines', 
		type=int, 
		default=1000000, 
		help='The number of SlimPajama lines to process.'
	)
	parser.add_argument(
		'--vocab_size',
		nargs='+',
		type=int,
		default=[8192, 14336, 16384, 32768, 65536],
		help='The vocab sizes to process.'
	)
	
	args = parser.parse_args()
	MODEL_PATH = args.model_path
	NUM_LINES = args.num_lines
	print(args)
	main(args)
