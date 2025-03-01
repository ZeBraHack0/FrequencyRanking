Question_File=data/human_eval/question.jsonl
Sample_File=/home/test/test01/pty/frspec/data/human_eval/model_answer/llama-3-8b-instruct/eagle-fr-spec-32768_correctness.jsonl

python evaluation/he_local/check_correctness.py \
    --question-file $Question_File \
    --sample-file $Sample_File