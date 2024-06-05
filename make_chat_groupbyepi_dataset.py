import re
import json
import argparse
from sklearn.model_selection import train_test_split
from openai import OpenAI
import os
from model_utils import model_response
from prompts import return_model_prompt, return_tag_prompt, return_extract_prompt, return_segment_prompt

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def get_gpt_response(prompt):

    try:
        response = client.chat.completions.create(
        model = "gpt-4",
        messages=[
            {
            "role": "user",
            "content": f"{prompt}"
            }
        ],
        temperature=0.9,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )

    except Exception as e:
        error_str = str(e)
        print("error: ", error_str)

        return None
    
    if response.choices:
        model_response = response.choices[0].message.content
        return model_response
    else:
        return None

def get_gpt_tag_response(prompt):
    
    try:
        response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": "You need to generate the next conversation based on the dialogue history. Before doing so, carefully review the candidates and choose the topic of the conversation wisely.\n"
                }
            ]
            },
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": "Based on the provided conversation history, identify three responses that would appropriately continue the dialogue. Choose the response labeled \"everyday language\" only if no other options are suitable.\n\nConversation History:\nsidney : (sidney shows an assertive personality and expresses his intentions of giving advice to susan like an uncle) wait for me ill be right back its not my nature susie but ill talk to you like an uncle susan : (susan displays independence and self-assuredness in declining sidney's offer to act as an uncle) but i dont need an uncle sidney sidney : (sidney demonstrates a certain level of admiration toward susan , sidney plays a role of mediator between susan and her brother, and possibly between susan and j.j. , sidney has plans to meet j.j. in the near future , susan has a love message for j.j. about steve dallas that she wants sidney to deliver) no i mean because i admire you in fact more than admire you although thats neither here nor there susie dont sell your brother short talk this over with him i mean youll find him a real friend any message in case i see jj later\nCandidate Responses:\n0. susan observes sidneys tendency to be touchy 1. susan is in love with a man named steve 2. susan displays independence and selfassuredness in declining sidneys offer to act as an uncle 3. susan is experiencing romantic feelings for the first time with a man named steve dallas 4. susan is jjs sister 5. susan is upset with both sidney and steve 6. susan is contemplating her future including her decision on whether to join steve for a long tour and her plan to discuss her relationship with steve with her brother jj the next morning she is also exercising introspection reflecting on her feelings towards her familial relationships and her romantic interest 7. susan desires for sidney and steve to get along 8. susan hints at a lack of selfworth when calling herself a worthless rag 9. susan seeks validation for her own identity separate from her brother 10. susan is considering embarking on an eightmonth tour with steve 11. susan is in a state of high emotional distress 12. susan is possibly unaware of a family members illness 13. susan is helpless and somewhat disorganized 14. everyday language 15. susan mentions giving up steve and regrets the loneliness of her brother suggesting a matured sense of responsibility 16. susan had contact with mr dangelo and steve 17. susan made a visit to the hospital 18. susan is distraught and feels guilt and sorrow indicating an emotional and sensitive nature 19. susan feels constricted by her identity as her brothers sister 20. susan may be younger or less experienced in relationships\n\nSelect the Appropriate Responses:\nPlease specify the responses in the following format:\nnumber1 : candidate response text\nnumber2: candidate response text\nnumber3 : candidate response text"
                }
            ]
            },
            {
            "role": "assistant",
            "content": [
                {
                "type": "text",
                "text": "1: susan is in love with a man named steve\n7: susan desires for sidney and steve to get along\n10: susan is considering embarking on an eight-month tour with steve"
                }
            ]
            },
            {
                "role" : "user",
                "content" : f"{prompt}"
            }
        ],
        temperature=0,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )

    except Exception as e:
        error_str =str(e)
        print("error:", error_str)
        return None
  
    # 모델의 텍스트 응답 추출
    if response.choices:
        model_response = response.choices[0].message.content
        return model_response
    else:
        return None
    

def get_gpt_segment_response(prompt):
  try:
    response = client.chat.completions.create(
    model="gpt-3.5-turbo-0125",
    messages=[
      {
        "role": "user",
        "content": "Please break down the following sentence into its core factual components without overly splitting the content. \n\n\"THUMPER is direct and confrontational, using strong language. He appears to be knowledgeable about psychodynamics, specifically repression.\"\n\nFor the output, list each cohesive factual unit with a number as follows:\n1. \n2.\n3.\n\nEnsure the breakdown retains natural phrasing while omitting any references to the significance, nature of the information, and discussions about the basis of any claims.\nReplace uncertain terms like \"appears\" or \"seems\" with more definitive expressions such as \"is\" to ensure the sentences convey clear and assertive information.\nMake sure to write in complete sentences and preserve the natural flow of information, excluding any explanations or justifications."
      },
      {
        "role": "assistant",
        "content": "1. THUMPER is direct and confrontational.\n2. THUMPER uses strong language.\n3. THUMPER is knowledgeable about psychodynamics. specifically repression."
      },
      {
        "role": "user",
        "content": "Please break down the following sentence into its core factual components without overly splitting the content. \n\n\"ADA seems to be inexperienced or lacks knowledge about farming and livestock, evident from her limited understanding of the uses of pigs besides hams.\"\n\nFor the output, list each cohesive factual unit with a number as follows:\n1. \n2.\n3.\n\nEnsure the breakdown retains natural phrasing while omitting any references to the significance, nature of the information, and discussions about the basis of any claims.\nReplace uncertain terms like \"appears\" or \"seems\" with more definitive expressions such as \"is\" to ensure the sentences convey clear and assertive information.\nMake sure to write in complete sentences and preserve the natural flow of information, excluding any explanations or justifications."
      },
      {
        "role": "assistant",
        "content": "1. ADA lacks knowledge about farming and livestock.\n2. ADA has a limited understanding of the uses of pigs besides hams."
      },
      {
        "role": "user",
        "content": "Please break down the following sentence into its core factual components without overly splitting the content. \n\n\"Anna's upcoming film, which begins shooting in L.A. on Tuesday, is her significant temporal information. \"\n\nFor the output, list each cohesive factual unit with a number as follows:\n1. \n2.\n3.\n\nEnsure the breakdown retains natural phrasing while omitting any references to the significance, nature of the information, and discussions about the basis of any claims.\nReplace uncertain terms like \"appears\" or \"seems\" with more definitive expressions such as \"is\" to ensure the sentences convey clear and assertive information.\nMake sure to write in complete sentences and preserve the natural flow of information, excluding any explanations or justifications."
      },
      {
        "role": "assistant",
        "content": "1. Anna's upcoming film begins shooting in L.A. on Tuesday."
      },
      {
        "role": "user",
        "content": "Please break down the following sentence into its core factual components without overly splitting the content. \n\n\"DOROTHY seems to have a threatening and unstable demeanor, and expresses a need for help.\"\n\nFor the output, list each cohesive factual unit with a number as follows:\n1. \n2.\n3.\n\nEnsure the breakdown retains natural phrasing while omitting any references to the significance, nature of the information, and discussions about the basis of any claims.\nReplace uncertain terms like \"appears\" or \"seems\" with more definitive expressions such as \"is\" to ensure the sentences convey clear and assertive information.\nMake sure to write in complete sentences and preserve the natural flow of information, excluding any explanations or justifications."
      },
      {
        "role": "assistant",
        "content": "1. DOROTHY has a threatening and unstable demeanor. \n2. DOROTHY expresses a need for help."
      },
      {
        "role": "user",
        "content": f"{prompt}"
      }
      ],
      temperature=0.3,
      max_tokens=1000,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )

  except Exception as e:
    error_str = str(e)
    print("error: ", error_str)

    return None
  
  if response.choices:
      model_response = response.choices[0].message.content
      return model_response
  else:
      return None

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
  attr_list = [f"{speaker1}'s persona", f"{speaker2}'s persona", "Shared memory"]
  
  p1 = []
  p2 = []
  shared = []

  for session in conversations:
    p1.extend(session[attr_list[0]])
    p2.extend(session[attr_list[1]])
    shared.extend(session[attr_list[2]])

  return p1, p2, shared


def make_tag_by_gpt_dataset(dialogue_history, candidates): #tag 추천 gpt-3.5
  candidates_string = "\n".join([f"{idx}. {value}" for idx, value in enumerate(candidates)])
  prompt = return_tag_prompt(dialogue_history, candidates_string)
  #print(prompt)
  print("Get GPT tag predict!")
  response = get_gpt_tag_response(prompt)
  gpt_choice = []

  if response:
    #print(response)
    # 각 줄을 분리하여 리스트로 변환
    try:
      lines = response.strip().split('\n')
      # 각 줄의 첫 번째 부분이 숫자인지 확인하고 숫자를 추출
      numbers = [int(line.split(':')[0]) for line in lines]
      #print("predict number:", numbers)  # 출력 예시: [2, 4, 8]
      gpt_choice = numbers
    except:
      print("GPT가 출력 형식을 잘못했어요!")

  else:
    print("GPT API ERROR")

  if len(gpt_choice) > 0:
    # 일단 tag 하나로 만들기
    gpt_candidate = candidates[gpt_choice[0]].rstrip('.')
  else:
    gpt_candidate = "Everyday Language"
  
  print(f"gpt predict: ", gpt_candidate)
  return gpt_candidate




def remove_sentences_with_phrase(text, phrase):
    # 문장 분리를 위한 정규 표현식 사용
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    
    # 주어진 문구를 포함하지 않는 문장만 필터링
    filtered_sentences = [sentence for sentence in sentences if phrase not in sentence]
    
    # 필터링된 문장들을 다시 하나의 문자열로 결합
    return ' '.join(filtered_sentences)

def after_extract_data(extract_informations, speaker1, speaker2):
  info_list = [f"{speaker1}'s persona", f"{speaker2}'s persona", f"{speaker1}'s temporal information", f"{speaker2}'s temporal information", "Shared memories", "Mutual events"]
  remove_sen = ["There is no", "There are no", "information is not", "information cannot be", "None", "No shared", "No temporal information", "no temporal information", "no information for", "no shared"]
            
  for info in info_list:
    for rem in remove_sen:
      if rem in extract_informations[info]:
        result_text = remove_sentences_with_phrase(extract_informations[info], rem)
        extract_informations[info] = result_text 
  
  for info in info_list:
    parts = extract_informations[info].split(":", 1)
    if len(parts) > 1:
       extract_informations[info] = parts[1].strip()
  
  return extract_informations

def extract_data_from_response(gpt_extract_response, speaker1, speaker2):
  info_list = [f"{speaker1}'s persona", f"{speaker2}'s persona", f"{speaker1}'s temporal information", f"{speaker2}'s temporal information", "Shared memories", "Mutual events"]
  extract_informations = {}
  for i, k in enumerate(info_list):
    extract_informations[k] = ''

  if gpt_extract_response:
    information = gpt_extract_response.split('***')
    if len(information) == 8:
      for i, k in enumerate(info_list):
          extract_informations[k] = information[i+1]

  print(extract_informations)
  extract_informations = after_extract_data(extract_informations, speaker1, speaker2)

  return extract_informations

def afterprocessing_extract_data(extract_data, speaker1, speaker2):
  attr_list = [f"{speaker1}'s persona", f"{speaker2}'s persona"]

  for att in attr_list:
    sentence = extract_data[att]
    if sentence == "":
      continue
    prompt = return_segment_prompt(sentence)
    model_response = get_gpt_segment_response(prompt)
    if model_response:
      
def make_dataset(data, outputfilename, flag):
  dataset = []

  count = 0 
  for key, value in data.items(): #episode당
    session_dataset = []
    speaker1, speaker2 = tuple(key.replace("(", "").replace(")", "").replace("'", "").split(", "))
    conversations = value['dialogue']
    print(speaker1, speaker2)

    speaker1_persona, speaker2_persona, shared_memory = return_first_memory_set(speaker1, speaker2, conversations)
    speaker1_temp = []
    speaker2_temp = []

    for session in conversations: #session 시작
      data_dic = {}
      dia_text = ""
      dia_no_tag_text = ""

      #dialogue
      if flag: #train
        dialogue_set = session['dialogues']
      else: #valid, test
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
      #utterance = model_response(prompt, tag, data_dic['last_speaker'])
      
      utterance = "This is just sentence."
      print(utterance)
      dia_no_tag_text += f"{data_dic['last_speaker']}: {utterance}\n"

      print(dia_text)
      print()
      print(dia_no_tag_text)
      extract_prompt = return_extract_prompt(speaker1, speaker2, dia_no_tag_text)
      gpt_extract_response = get_gpt_response(extract_prompt)
      extract_data = extract_data_from_response(gpt_extract_response, speaker1, speaker2)
      print(extract_data)
      extract_data = afterprocessing_extract_data(extract_data, speaker1, speaker2)
      assert False

      
      session_dataset.append(data_dic)
    dataset.append(session_dataset)

    count += 1
      
  with open(outputfilename, 'w', encoding='utf-8') as f:
      for item in dataset:
          json.dump(item, f, ensure_ascii=False)
          f.write('\n')  # 각 딕셔너리 후 줄바꿈 추가



# main function

# python make_chat_gr_dataset.py final_list_dataset.json

parser = argparse.ArgumentParser(description='json to excel file')
parser.add_argument('json', type=str, help='The json file')
args = parser.parse_args()

my_json_file = args.json

with open(my_json_file, 'r') as file:
  data = json.load(file)


print(len(data))
# 데이터 분할
train_data, valid_data, test_data = split_data(data, random_state=42)

print("Train Data:", len(train_data))
print("Validation Data:", len(valid_data))
print("Test Data:", len(test_data))

#make_dataset(train_data, "train_without_tag.json", 1)
#make_dataset(valid_data, "valid_without_tag.json", 0)
make_dataset(test_data, "test_without_tag.json", 0)