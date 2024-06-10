import os
import math
import torch
from transformers import (
    AutoConfig,
)
from transformers.deepspeed import HfDeepSpeedConfig

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
    PeftConfig,
    prepare_model_for_kbit_training,
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaForCausalLM,
    BitsAndBytesConfig,
)


def generate(
    model,
    tokenizer,
    inputs,
    num_beams=1,
    num_beam_groups=1,
    do_sample=True,
    num_return_sequences=1,
    max_new_tokens=100,
):
    generate_ids = model.generate(
        inputs.input_ids,
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new_tokens,
    )

    result = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return result


def get_peft_llama(path, device):
    print(path)
    config = PeftConfig.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name_or_path, fast_tokenizer=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path, load_in_8bit=True
    )
    model = PeftModel.from_pretrained(model, path)

    model.to(device)
    model.eval()

    return model, tokenizer


def get_peft_gemma(path, device):
    print(path)
    config = PeftConfig.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name_or_path, fast_tokenizer=True
    )
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, path)

    model.to(device)
    model.eval()

    return model, tokenizer


def model_response(prompt, tag, next_speaker, model, tokenizer, device):
    last_utter = f"{next_speaker}: ({tag})"
    dialogue = prompt.rstrip() + "\n" + last_utter
    input_ = tokenizer(dialogue, return_tensors="pt").to(device)
    output = generate(
        model,
        tokenizer,
        input_,
        num_beams=1,
        num_return_sequences=1,
        max_new_tokens=200,
    )
    ### utterance : correct answer
    ### response : Model-generated answer
    response = output.replace(dialogue, "").split("\n")[0].strip()

    return response
