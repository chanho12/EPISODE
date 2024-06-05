import os
import math
import torch
from transformers import (
    AutoConfig,
)
from transformers.deepspeed import HfDeepSpeedConfig

from peft import LoraConfig, TaskType, get_peft_model, PeftModel, PeftConfig, prepare_model_for_kbit_training

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaForCausalLM,
    BitsAndBytesConfig
)

def generate(model,
             tokenizer,
             inputs,
             num_beams=1,
             num_beam_groups=1,
             do_sample=True,
             num_return_sequences=1,
             max_new_tokens=100):

    generate_ids = model.generate(inputs.input_ids,
                                  num_beams=num_beams,
                                  num_beam_groups=num_beam_groups,
                                  do_sample=do_sample,
                                  num_return_sequences=num_return_sequences,
                                  max_new_tokens=max_new_tokens)

    result = tokenizer.batch_decode(generate_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)[0]
    
    return result

def get_peft_llama(path, device):
	print(path)
	config = PeftConfig.from_pretrained(path)
	tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path,fast_tokenizer=True)
	model= AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, load_8bits = True)
	model = PeftModel.from_pretrained(model, path)

	model.to(device)
	model.eval()

	return model, tokenizer

def get_peft_gemma(path, device):
	print(path)
	config = PeftConfig.from_pretrained(path)
	tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path,fast_tokenizer=True)
	model= AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
	model = PeftModel.from_pretrained(model, path)

	model.to(device)
	model.eval()

	return model, tokenizer




def model_response(prompt, tag, next_speaker,model, tokenizer,device):

  last_utter = f"{next_speaker} : ({tag}) "
  dialogue = prompt + last_utter

  input_ = tokenizer(dialogue, return_tensors = 'pt').to(device)
  output = generate(model,tokenizer,
                                input_,
                                num_beams=1,
                                num_return_sequences=1,
                                max_new_tokens=100)


  ### utterance : correct answer
  ### response : Model-generated answer
  response = output.replace(dialogue, '').split("\n")[0]

  return response


prompt = {"prompt": "\nTask: Generate the next response in a dialogue by focusing on the contextual cues detailed within parentheses in the dialogue history. Responses should be tailored according to the type of cue provided:\n\n1. Memory-driven dialogues: If the cue within parentheses details specific character traits or background context, craft responses that reflect these memory-driven elements, ensuring character consistency and rich context.\n2. Everyday language dialogues: If the cue within parentheses is labeled \"Everyday Language,\" generate responses that are based on typical day-to-day interactions, free from specific personas or detailed context.\n\n**Dialogue History**:\nLOLA: (Everyday Language) Hello, Mr. Neff. It's me.\nNEFF: (Everyday Language) Something the matter?\nLOLA: (LOLA has been waiting for NEFF) I've been waiting for you.\nNEFF: (Everyday Language) For me? What for?\nLOLA: (Everyday Language) I thought you could let me ride with you, if you're going my way.\n\n", "answer": "Which way would that be? Oh, sure. Vermont and Franklin. North- west corner, wasn't it? Be glad to, Miss Dietrichson.", "gold_tag": "NEFF is familiar with the local geographic area , NEFF references specific streets", "last_speaker": "NEFF"}
path = 'chano12/gemma_with_tag'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer = get_peft_gemma(path, device)

response = model_response(prompt['prompt'], prompt['gold_tag'], prompt['last_speaker'], model, tokenizer, device)
print(response)