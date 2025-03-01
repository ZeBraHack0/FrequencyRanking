tokenizer_path="meta-llama/Llama-3.2-1B-instruct"
Vocab=32768

baseline="/home/test/test01/pty/llamacu/data/gsm8k/model_answer/llama-3-8b-instruct/baseline.jsonl"
eagle_original="/home/test/test01/pty/llamacu/data/gsm8k/model_answer/llama-3-8b-instruct/eagle-original.jsonl"
eagle_fr_spec="/home/test/test01/pty/llamacu/data/gsm8k/model_answer/llama-3-8b-instruct/eagle-slimpajama-32768.jsonl"

echo "EAGLE ORIGINAL"
python evaluation/human_eval/speed.py \
    --file-path $eagle_original \
    --base-path $baseline \
    --checkpoint-path $tokenizer_path

echo "EAGLE FR-SPEC"
python evaluation/human_eval/speed.py \
    --file-path $eagle_fr_spec \
    --base-path $baseline \
    --checkpoint-path $tokenizer_path