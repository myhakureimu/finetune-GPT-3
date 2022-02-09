#!/bin/bash
model=$(./model_select_gen.sh $1)
cd data
cat test_prompt_$1|{
    while read line;do
        openai api completions.create -m $model -p "$line" >> test_completion_$1
        echo "\n">>test_completion_$1
    done
}
cd ..