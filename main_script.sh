#!/bin/bash
python main.py \
--path='chano12/llama_with_tag' \
--tag_type='gpt' \
--update_method='update' \
--input_file=list_dataset.json \
--output_file='gpttag_llama_accumulate.json'
