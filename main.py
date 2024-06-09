import re
import json
import argparse
from sklearn.model_selection import train_test_split
from model_utils import model_response, get_peft_gemma, get_peft_llama
from data_extract import (
    get_gpt_response,
    return_extract_prompt,
    extract_data_from_response,
)
from gpt_tag_predict import make_tag_by_gpt_dataset
import torch


def remove_parentheses(text):
    pattern = r"\([^()]*\)"
    while re.search(pattern, text):
        text = re.sub(pattern, "", text)
    return text.strip()


def split_data(data, train_size=0.8, valid_size=0.1, test_size=0.1, random_state=None):
    assert train_size + valid_size + test_size == 1, "합은 1이어야 합니다."

    keys = list(data.keys())
    values = list(data.values())

    train_keys, remaining_keys, train_values, remaining_values = train_test_split(
        keys, values, train_size=train_size, random_state=random_state
    )

    valid_keys, test_keys, valid_values, test_values = train_test_split(
        remaining_keys,
        remaining_values,
        train_size=valid_size / (valid_size + test_size),
        random_state=random_state,
    )

    train_data = dict(zip(train_keys, train_values))
    valid_data = dict(zip(valid_keys, valid_values))
    test_data = dict(zip(test_keys, test_values))

    return train_data, valid_data, test_data


def return_first_memory_set(conversations):
    # 오직 shared memory만
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


def memory_update(
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
        if extract_data[key] != "":
            target_list.extend(extract_data[key])

    extend_list_if_not_empty(f"{speaker1}'s persona", speaker1_persona)
    extend_list_if_not_empty(f"{speaker2}'s persona", speaker2_persona)
    extend_list_if_not_empty(f"{speaker1}'s temporal information", speaker1_temp)
    extend_list_if_not_empty(f"{speaker2}'s temporal information", speaker2_temp)
    extend_list_if_not_empty("Shared memories", shared_memory)
    extend_list_if_not_empty("Mutual events", shared_memory)

    return (
        speaker1_persona,
        speaker2_persona,
        speaker1_temp,
        speaker2_temp,
        shared_memory,
    )


def make_dialouges(session):
    dia_text = ""
    dia_no_tag_text = ""
    dialogue_set = session["dialogues"][:-1]

    for dial in dialogue_set:
        dia_text += (
            f"{dial['speaker']}: ({remove_parentheses(dial['label'])}) {dial['text']}\n"
        )
    for dial in dialogue_set:
        dia_no_tag_text += f"{dial['speaker']}: {dial['text']}\n"

    return dia_text, dia_no_tag_text


def load_data(path):
    with open(path, "r") as file:
        data = json.load(file)

    filtered_data = {}
    for key, value in data.items():
        if "dialogue" in value and len(value["dialogue"]) >= 5:
            filtered_data[key] = value

    return filtered_data


def episode(data, tag_type, update_method, outputfilename, model, tokenizer, device):
    dataset = []
    count = 0

    for key, value in data.items():  # episode
        session_dataset = []
        speaker1, speaker2 = tuple(
            key.replace("(", "").replace(")", "").replace("'", "").split(", ")
        )
        conversations = value["dialogue"]

        shared_memory = return_first_memory_set(conversations)
        speaker1_persona = []
        speaker2_persona = []
        speaker1_temp = []
        speaker2_temp = []

        for session in conversations[:5]:  # session
            data_dic = {}

            dia_text, dia_no_tag_text = make_dialouges(session)

            last = session["dialogues"][-1]
            prompt = return_model_prompt(dia_text)
            data_dic["answer"] = last["text"]
            last_speaker = last["speaker"]
            data_dic["last_speaker"] = last_speaker
            data_dic["prompt"] = prompt

            if last_speaker == speaker1:
                candidates = speaker1_persona + speaker1_temp + shared_memory
            else:
                candidates = speaker2_persona + speaker2_temp + shared_memory

            tag = remove_parentheses(last["label"])
            # gpt tag predict
            if tag_type == "gpt":
                tag = make_tag_by_gpt_dataset(dia_text, candidates)
            data_dic["tag"] = tag

            # model inference
            utterance = model_response(
                prompt, tag, last_speaker, model, tokenizer, device
            )
            data_dic["model_response"] = utterance
            dia_no_tag_text += f"{last_speaker}: {utterance}\n"

            # model extract
            extract_prompt = return_extract_prompt(speaker1, speaker2, dia_no_tag_text)
            gpt_extract_response = get_gpt_response(extract_prompt)
            extract_data = extract_data_from_response(
                gpt_extract_response, speaker1, speaker2
            )

            # model update
            if update_method == "accumulate":
                (
                    speaker1_persona,
                    speaker2_persona,
                    speaker1_temp,
                    speaker2_temp,
                    shared_memory,
                ) = memory_update(
                    extract_data,
                    speaker1,
                    speaker2,
                    speaker1_persona,
                    speaker2_persona,
                    speaker1_temp,
                    speaker2_temp,
                    shared_memory,
                )

            # debugging
            # print(f"{speaker1}'s persona: {speaker1_persona}")
            # print(f"{speaker2}'s persona: {speaker2_persona}")
            # print(f"{speaker1}'s tem: {speaker1_temp}")
            # print(f"{speaker2}'s tem: {speaker2_temp}")
            # print(f"shared memory: {shared_memory}")

            # save
            session_dataset.append(data_dic)

        dataset.append(session_dataset)

        count += 1

    with open(outputfilename, "w", encoding="utf-8") as f:
        for item in dataset:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")  # 각 딕셔너리 후 줄바꿈 추가


# main function

# python main.py --path='chano12/llama_with_tag' --tag_type='gold' --update_method='accumulate' --input_file=list_dataset.json --output_file='goldtag_llama_accumulate.json'
# python main.py --path='chano12/llama_with_tag' --tag_type='gpt' --update_method='accumulate' --input_file=list_dataset.json --output_file='gpttag_llama_accumulate.json'


def main():
    parser = argparse.ArgumentParser(description="json to excel file")
    parser.add_argument("--path", type=str, help="model path")
    parser.add_argument("--tag_type", type=str, help="model path")
    parser.add_argument("--update_method", type=str, help="model path")
    parser.add_argument("--input_file", type=str, help="The input file")
    parser.add_argument("--output_file", type=str, help="The output file")
    args = parser.parse_args()

    my_json_file = args.input_file
    my_output_json_file = args.output_file
    path = args.path
    tag_type = args.tag_type
    update_method = args.update_method

    filtered_data = load_data(my_json_file)
    # 필터링된 결과 출력
    print(len(filtered_data))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if path == "chano12/llama_with_tag":
        model, tokenizer = get_peft_llama(path, device)
    else:
        model, tokenizer = get_peft_gemma(path, device)

    # split data
    train_data, valid_data, test_data = split_data(filtered_data, random_state=42)

    print("Train Data:", len(train_data))
    print("Validation Data:", len(valid_data))
    print("Test Data:", len(test_data))

    episode(
        test_data,
        tag_type,
        update_method,
        my_output_json_file,
        model,
        tokenizer,
        device,
    )


if __name__ == "__main__":
    main()
