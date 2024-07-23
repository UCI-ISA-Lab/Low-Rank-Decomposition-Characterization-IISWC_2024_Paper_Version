#!/bin/bash

random_number=$RANDOM
orig_model_path="../Llama-2-7B/Llama-2-7b-chat-hf/"
new_path="../Llama-2-7B/Llama-2-7b-chat-hf-$random_number/"
mv $orig_model_path $new_path
output_path="output.jsonl"
rm -rf $output_path
accelerate launch -m lm_eval --model hf --model_args pretrained=$new_path --tasks $1 --output_path $output_path --batch_size auto 2>&1 | tee terminal_output.log
mv $new_path $orig_model_path