import re
import json
import argparse
from sklearn.model_selection import train_test_split
import os
from model_utils import model_response, get_peft_gemma, get_peft_llama
from data_extract import get_gpt_response, return_extract_prompt, extract_data_from_response
from gpt_tag_predict import make_tag_by_gpt_dataset
import torch


def remove_parentheses(text):
    # Regex pattern to remove text within parentheses (handles nested parentheses by iterative removal)
    pattern = r'\([^()]*\)'
    # Iteratively remove all nested parentheses
    while re.search(pattern, text):
        text = re.sub(pattern, '', text)
    return text.strip()

def split_data(data, train_size=0.8, valid_size=0.1, test_size=0.1, random_state=None):
    assert train_size + valid_size + test_size == 1, "합은 1이어야 합니다."

    keys = list(data.keys())
    values = list(data.values())

    # 데이터 섞기 및 train/valid+test로 첫 번째 분할
    train_keys, remaining_keys, train_values, remaining_values = train_test_split(
        keys, values, train_size=train_size, random_state=random_state)

    # valid+test를 valid/test로 두 번째 분할
    valid_keys, test_keys, valid_values, test_values = train_test_split(
        remaining_keys, remaining_values, train_size=valid_size/(valid_size + test_size), random_state=random_state)

    train_data = dict(zip(train_keys, train_values))
    valid_data = dict(zip(valid_keys, valid_values))
    test_data = dict(zip(test_keys, test_values))

    return train_data, valid_data, test_data

def return_first_memory_set(speaker1, speaker2, conversations):
  #오직 shared memory만 
  shared = []

  for session in conversations:
    shared.extend(session["Shared memory"])

  return shared


def return_model_prompt(dia_text):
  prompt = f"""
Task: Generate the next response in a dialogue by focusing on the contextual cues detailed within parentheses in the dialogue history. Responses should be tailored according to the type of cue provided:

1. Memory-driven dialogues: If the cue within parentheses details specific character traits or background context, craft responses that reflect these memory-driven elements, ensuring character consistency and rich context.
2. Everyday language dialogues: If the cue within parentheses is labeled "Everyday Language," generate responses that are based on typical day-to-day interactions, free from specific personas or detailed context.

**Dialogue History**:
{dia_text}

"""
  return prompt



def memory_update(extract_data, speaker1, speaker2, speaker1_persona, speaker2_persona, speaker1_temp, speaker2_temp, shared_memory):
   
  # accumulate
  def extend_list_if_not_empty(key, target_list):
      if extract_data[key] != "":
          target_list.extend(extract_data[key])

  # 함수를 사용하여 데이터를 추가합니다.
  extend_list_if_not_empty(f"{speaker1}'s persona", speaker1_persona)
  extend_list_if_not_empty(f"{speaker2}'s persona", speaker2_persona)
  extend_list_if_not_empty(f"{speaker1}'s temporal information", speaker1_temp)
  extend_list_if_not_empty(f"{speaker2}'s temporal information", speaker2_temp)
  extend_list_if_not_empty("Shared memories", shared_memory)
  extend_list_if_not_empty("Mutual events", shared_memory)

  return speaker1_persona, speaker2_persona, speaker1_temp, speaker2_temp, shared_memory

def episode(data, outputfilename, model, tokenizer, device):
  dataset = []

  count = 0 
  for key, value in data.items(): #episode당
    session_dataset = []
    speaker1, speaker2 = tuple(key.replace("(", "").replace(")", "").replace("'", "").split(", "))
    conversations = value['dialogue']

    shared_memory = return_first_memory_set(speaker1, speaker2, conversations)
    speaker1_persona = []
    speaker2_persona = []
    speaker1_temp = []
    speaker2_temp = []

    for session in conversations[:5]: #session 시작
      data_dic = {}
      dia_text = ""
      dia_no_tag_text = ""

      dialogue_set = session['dialogues'][:-1]
      for dial in dialogue_set:
        dia_text += f"{dial['speaker']}: ({remove_parentheses(dial['label'])}) {dial['text']}\n"
      for dial in dialogue_set:
        dia_no_tag_text += f"{dial['speaker']}: {dial['text']}\n"

      dia = session['dialogues'][-1]
      prompt = return_model_prompt(dia_text)
      data_dic['prompt'] = prompt
      data_dic['answer'] = dia['text']
      data_dic['last_speaker'] = dia['speaker']

      if data_dic['last_speaker'] == speaker1:
        candidates = speaker1_persona + shared_memory + speaker1_temp
      else:
        candidates = speaker2_persona + shared_memory + speaker2_temp
      #gpt tag predict
      data_dic['gpt_tag'] = make_tag_by_gpt_dataset(dia_text, candidates)
      #gold tag
      data_dic['gold_tag'] = remove_parentheses(dia['label'])
      
      tag = data_dic['gpt_tag']
      #model inference
      utterance = model_response(prompt, tag, data_dic['last_speaker'], model, tokenizer, device)
      data_dic['model_response'] = utterance
      #utterance = "This is just sentence."
      dia_no_tag_text += f"{data_dic['last_speaker']}: {utterance}\n"
      print(dia_no_tag_text)

      extract_prompt = return_extract_prompt(speaker1, speaker2, dia_no_tag_text)
      gpt_extract_response = get_gpt_response(extract_prompt)
      extract_data = extract_data_from_response(gpt_extract_response, speaker1, speaker2)

      speaker1_persona, speaker2_persona, speaker1_temp, speaker2_temp, shared_memory = memory_update(extract_data, speaker1, speaker2, speaker1_persona, speaker2_persona, speaker1_temp, speaker2_temp, shared_memory)
      print(extract_data)
      print(speaker1_persona, speaker2_persona, speaker1_temp, speaker2_temp, shared_memory)
      
      # save
      session_dataset.append(data_dic)

    dataset.append(session_dataset)

    count += 1
      
  with open(outputfilename, 'w', encoding='utf-8') as f:
      for item in dataset:
          json.dump(item, f, ensure_ascii=False)
          f.write('\n')  # 각 딕셔너리 후 줄바꿈 추가



# main function

# python make_chat_groupbyepi_dataset.py final_list_dataset.json

parser = argparse.ArgumentParser(description='json to excel file')
parser.add_argument('json', type=str, help='The json file')
args = parser.parse_args()


my_json_file = args.json

with open(my_json_file, 'r') as file:
  data = json.load(file)


# 결과를 저장할 딕셔너리
filtered_data = {}

# 각 키에 대해 반복
for key, value in data.items():
    
    if 'dialogue' in value and len(value['dialogue']) >= 5:
        filtered_data[key] = value

# 필터링된 결과 출력
print(len(filtered_data))

path = 'chano12/gemma_with_tag'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer = get_peft_gemma(path, device)

# 데이터 분할
train_data, valid_data, test_data = split_data(filtered_data, random_state=42)

print("Train Data:", len(train_data))
print("Validation Data:", len(valid_data))
print("Test Data:", len(test_data))

#make_dataset(train_data, "train_without_tag.json", 1)
#make_dataset(valid_data, "valid_without_tag.json", 0)
#episode(test_data, "test_without_tag.json", model, tokenizer, device)