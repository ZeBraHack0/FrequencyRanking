Question_File=data/human_eval/question.jsonl
Sample_File=/home/test/test01/pty/llamacu/data/human_eval/model_answer/llama-3-8b-instruct/baseline-new-prompt_correctness.jsonl

python evaluation/human_eval/check_correctness.py \
    --question-file $Question_File \
    --sample-file $Sample_File