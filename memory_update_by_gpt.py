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


def get_gpt_response(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[{"role": "user", "content": f"{prompt}"}],
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


def return_prompt(dialogues_text, candidates):
    prompt = f""" 

"""
    return prompt


def memory_update(data):
    count = 0
    for key, value in data.items():
        count += 1
        if count > 5:
            break
        speaker1, speaker2 = tuple(
            key.replace("(", "").replace(")", "").replace("'", "").split(", ")
        )
        conversations = value["dialogue"]

        for idx, session in enumerate(conversations):
            if idx == 0:
                continue
            previous_memory = ""
            current_memory = ""
            attribute = [
                f"{speaker1}'s persona",
                f"{speaker1}'s temporary event",
                f"{speaker2}'s persona",
                f"{speaker2}'s temporary event",
            ]

            for att in attribute:
                previous_att = value["dialogue"][idx - 1][att]
                current_att = session[att]
                if len(previous_att) > 0:
                    previous_memory += f"- {att}: \n"
                for pre in previous_att:
                    previous_memory += pre + "\n"
                if len(current_att) > 0:
                    current_memory += f"- {att} : \n"
                for cur in current_att:
                    current_memory += cur + "\n"

            print(previous_memory)
            print("-" * 100)
            print(current_memory)
            print("^" * 100)


###main###
def main():
    file_name = "list_dataset.json"
    data = load_data(file_name)
    memory_update(data)


if __name__ == "__main__":
    main()
