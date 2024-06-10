import torch
import json
import re
from openai import OpenAI
import argparse
import random
import os

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


def load_data(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def get_gpt_response(message):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=message,
            temperature=0,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

    except Exception as e:
        error_str = str(e)
        print("error:", error_str)
        return None

    # 모델의 텍스트 응답 추출
    if response.choices:
        model_response = response.choices[0].message.content
        return model_response
    else:
        return None

def mutual_to_shared(text):
    messages=[
            {
            "role": "system",
            "content": "Given the following sentences in present continuous tense, please convert them to past tense. Make sure the converted sentences accurately reflect completed actions."
            },
            {
                "role": "user",
                "content" : f"{text}"
            }
            ]
    return messages

def return_prompt(previous_memory, current_memory):
    prompt = f"""\
You are an AI model tasked with updating the memory concerning two individuals based on provided previous and current memories. 
The update should reflect semantic similarities and differences between entries in the memories.

Instructions for Memory Update:
1. Analyze Semantically Similar Sentences:
When sentences in both the previous and current memories express similar meanings or ideas, prioritize the sentence from the current memory. This adjustment ensures that the memory reflects the most up-to-date expressions and nuances of the situation or sentiment.
2. Handle Contrasting or Contradictory Sentences:
When encountering sentences that express opposite sentiments or conflicting information between the previous and current memories, always select the sentence from the current memory. This ensures that the memory reflects the most recent developments or changes in sentiment.
3. Manage Unrelated Sentences:
If sentences in the previous and current memories are unrelated, include both in the updated memory to preserve the full spectrum of information.
4. Avoid Introducing New Information:
Do not add any new information that does not appear in either the previous or current memories. Ensure that all entries in the updated memory are derived solely from the provided data.

Example:
Previous Memory:
John enjoys discussing his future plans and has a passion for his work. 
John loves running marathons and has been training for the New York Marathon. 
John has always had a strong bond with his family.

Current Memory:
Recently, John has been feeling uncertain about his career direction and mentioned taking up gardening as a new hobby. 
John decided to skip the New York Marathon this year due to a knee injury. 
John relationship with his family has been strained due to work pressure.

Updated Memory:
John has been feeling uncertain about his career direction recently and enjoys discussing his future plans. 
John mentioned taking up gardening as a new hobby. 
Despite his passion for running, John decided to skip the New York Marathon this year due to a knee injury. 
Although John generally has a strong bond with his family, recent work pressures have strained their relationship.

Task:
Previous Memory:
{previous_memory}

Current Memory:
{current_memory}

Updated Memory:"""

    message = [
            {
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": "You are an AI model tasked with updating the memory regarding two individuals."
                }
            ]
            },
            {
                "role": "user", 
                "content": f"{prompt}"
            }
            ]
    print("*"*100)
    print("Prompt\n:", prompt)
    print("*"*100)
    return message


def return_sentences(text):
    # 숫자. 없애기
    text = re.sub(r"\d+\.\s*", "", text)
    # 줄 바꿈 문자를 공백으로 대체합니다.
    text = text.replace("\n", " ")
    # 중복된 공백을 하나의 공백으로 줄입니다.
    text = re.sub(r"\s+", " ", text)
    # 예외케이스
    text = re.sub(r"\b(Mr|Mrs|Ms|Dr|Jr|Sr|Prof)\.", r"\1<dot>", text)
    # 정규 표현식을 사용하여 문장을 분리
    sentence_endings = re.compile(r"(?<=\.|\?|!)\s")
    sentences = sentence_endings.split(text.strip())
    # 약어의 점을 원래대로 복원
    sentences = [sentence.replace("<dot>", ".") for sentence in sentences]
    # 빈 문자열을 제거

    return sentences

def memory_gpt_update(
    extract_data,
    speaker1,
    speaker2,
    speaker1_persona,
    speaker2_persona,
    speaker1_temp,
    speaker2_temp,
    shared_memory,
):
    # accumulate
    def extend_list_if_not_empty(key, target_list):
        if extract_data[key] != "": #current
            if len(target_list) == 0: #previous
                return extract_data[key]
            previous_string = "\n".join(target_list)
            current_string = "\n".join(extract_data[key])
            message = return_prompt(previous_string, current_string)
            response = get_gpt_response(message)
            print("\nGPT response:",response)
            update_memory = return_sentences(response)
            return update_memory
        else:
            return target_list
    
    def update_mutual(key, target_list):
        if extract_data[key] != "":
            current_string = "\n".join(extract_data[key])
            message = mutual_to_shared(current_string)
            response = get_gpt_response(message)
            update_memory = return_sentences(response)
            if update_memory:
                target_list.extend(update_memory)
                return target_list
        else:
            return target_list


    speaker1_persona = extend_list_if_not_empty(f"{speaker1}'s persona", speaker1_persona)
    speaker2_persona = extend_list_if_not_empty(f"{speaker2}'s persona", speaker2_persona)
    speaker1_temp = extend_list_if_not_empty(f"{speaker1}'s temporal information", speaker1_temp)
    speaker2_temp = extend_list_if_not_empty(f"{speaker2}'s temporal information", speaker2_temp)
    shared_memory = extend_list_if_not_empty("Shared memories", shared_memory)
    shared_memory = update_mutual("Mutual events", shared_memory)

    return (
        speaker1_persona,
        speaker2_persona,
        speaker1_temp,
        speaker2_temp,
        shared_memory,
    )


###main###
def main():
    extract_data = {"ACHILLES's persona": ['ACHILLES is a warrior who is indifferent to his job.', 'ACHILLES likes dogs more than people.', 'ACHILLES is interested in the concept of mortality and believes that gods envy humans for it.', 'ACHILLES enjoys provoking BRISEIS.', 'ACHILLES believes that gods, including Zeus, Athena, and Aries, envy the mortality of humans.'], "BRISEIS's persona": ['BRISEIS is brave and fights back when attacked.', 'BRISEIS has dedicated her life to the gods and serves Zeus, Athena, and Aries.', 'BRISEIS believes that all gods are to be feared and respected.', "BRISEIS questions ACHILLES' choice of being a warrior and thinks he enjoys it.", "BRISEIS is aware of the motives of ACHILLES' visit to Troy.", 'BRISEIS possibly feels for ACHILLES, given his remark about her being a woman in love with a god.'], "ACHILLES's temporal information": '', "BRISEIS's temporal information": '', 'Shared memories': '', 'Mutual events': ['ACHILLES and BRISEIS are engaged in a deep conversation about their beliefs, roles, and gods.', "ACHILLES questions BRISEIS' devotion to the gods, and BRISEIS questions ACHILLES' choice of being a warrior."]}
    speaker1 = 'ACHILLES'
    speaker2 = 'BRISEIS'
    speaker1_persona = ['ACHILLES has sweet boy friend.']
    speaker2_persona = []
    speaker1_temp = []
    speaker2_temp = []
    shared_memory = []
    print(f"{speaker1}'s persona: {speaker1_persona}")
    print(f"{speaker2}'s persona: {speaker2_persona}")
    print(f"{speaker1}'s tem: {speaker1_temp}")
    print(f"{speaker2}'s tem: {speaker2_temp}")
    print(f"shared memory: {shared_memory}")
    (
        speaker1_persona,
        speaker2_persona,
        speaker1_temp,
        speaker2_temp,
        shared_memory,
    ) = memory_gpt_update(
        extract_data,
        speaker1,
        speaker2,
        speaker1_persona,
        speaker2_persona,
        speaker1_temp,
        speaker2_temp,
        shared_memory,
    )
    print(f"{speaker1}'s persona: {speaker1_persona}")
    print(f"{speaker2}'s persona: {speaker2_persona}")
    print(f"{speaker1}'s tem: {speaker1_temp}")
    print(f"{speaker2}'s tem: {speaker2_temp}")
    print(f"shared memory: {shared_memory}")


if __name__ == "__main__":
    main()
