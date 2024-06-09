import os
from openai import OpenAI
import re

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


def get_gpt_response(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": f"{prompt}"}],
            temperature=0.9,
            max_tokens=300,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
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


def get_gpt_segment_response(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {
                    "role": "user",
                    "content": 'Please break down the following sentence into its core factual components without overly splitting the content. \n\n"THUMPER is direct and confrontational, using strong language. He appears to be knowledgeable about psychodynamics, specifically repression."\n\nFor the output, list each cohesive factual unit with a number as follows:\n1. \n2.\n3.\n\nEnsure the breakdown retains natural phrasing while omitting any references to the significance, nature of the information, and discussions about the basis of any claims.\nReplace uncertain terms like "appears" or "seems" with more definitive expressions such as "is" to ensure the sentences convey clear and assertive information.\nMake sure to write in complete sentences and preserve the natural flow of information, excluding any explanations or justifications.',
                },
                {
                    "role": "assistant",
                    "content": "1. THUMPER is direct and confrontational.\n2. THUMPER uses strong language.\n3. THUMPER is knowledgeable about psychodynamics. specifically repression.",
                },
                {
                    "role": "user",
                    "content": 'Please break down the following sentence into its core factual components without overly splitting the content. \n\n"ADA seems to be inexperienced or lacks knowledge about farming and livestock, evident from her limited understanding of the uses of pigs besides hams."\n\nFor the output, list each cohesive factual unit with a number as follows:\n1. \n2.\n3.\n\nEnsure the breakdown retains natural phrasing while omitting any references to the significance, nature of the information, and discussions about the basis of any claims.\nReplace uncertain terms like "appears" or "seems" with more definitive expressions such as "is" to ensure the sentences convey clear and assertive information.\nMake sure to write in complete sentences and preserve the natural flow of information, excluding any explanations or justifications.',
                },
                {
                    "role": "assistant",
                    "content": "1. ADA lacks knowledge about farming and livestock.\n2. ADA has a limited understanding of the uses of pigs besides hams.",
                },
                {
                    "role": "user",
                    "content": 'Please break down the following sentence into its core factual components without overly splitting the content. \n\n"Anna\'s upcoming film, which begins shooting in L.A. on Tuesday, is her significant temporal information. "\n\nFor the output, list each cohesive factual unit with a number as follows:\n1. \n2.\n3.\n\nEnsure the breakdown retains natural phrasing while omitting any references to the significance, nature of the information, and discussions about the basis of any claims.\nReplace uncertain terms like "appears" or "seems" with more definitive expressions such as "is" to ensure the sentences convey clear and assertive information.\nMake sure to write in complete sentences and preserve the natural flow of information, excluding any explanations or justifications.',
                },
                {
                    "role": "assistant",
                    "content": "1. Anna's upcoming film begins shooting in L.A. on Tuesday.",
                },
                {
                    "role": "user",
                    "content": 'Please break down the following sentence into its core factual components without overly splitting the content. \n\n"DOROTHY seems to have a threatening and unstable demeanor, and expresses a need for help."\n\nFor the output, list each cohesive factual unit with a number as follows:\n1. \n2.\n3.\n\nEnsure the breakdown retains natural phrasing while omitting any references to the significance, nature of the information, and discussions about the basis of any claims.\nReplace uncertain terms like "appears" or "seems" with more definitive expressions such as "is" to ensure the sentences convey clear and assertive information.\nMake sure to write in complete sentences and preserve the natural flow of information, excluding any explanations or justifications.',
                },
                {
                    "role": "assistant",
                    "content": "1. DOROTHY has a threatening and unstable demeanor. \n2. DOROTHY expresses a need for help.",
                },
                {"role": "user", "content": f"{prompt}"},
            ],
            temperature=0.3,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
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


def return_extract_prompt(speaker1, speaker2, dialogues_text):
    prompt = f"""You are a conversation analyst tasked with examining two conversations.

In your analysis, categorize the dialogue based on five criteria:
1. **Persona Information**: Discuss aspects such as personality, job, age, education, favorite foods, music, hobbies, family life, daily activities, health, etc.
2. **Temporal information**: Identify information that will soon become irrelevant, such as upcoming deadlines like "I need to submit my assignment by Friday" or temporary states like "I have a cold."
3. **Shared Memory**: Focus on past experiences that the speakers refer to during their conversation, which they have previously experienced together. This category includes both explicitly mentioned memories and those implied through their dialogue. For example, the exchange 'Alice: Wasn’t that jazz festival we went to last summer amazing?' 'Bob: It was phenomenal, especially the live band under the stars.' should be categorized here because it indicates that Alice and Bob shared the experience of attending a jazz festival together.
4. **Mutual Event**: This category captures significant events and interactions occurring directly between {speaker1} and {speaker2} during the current conversation, excluding any third-party involvement. Consider only those interactions that are substantial and directly involve both speakers. For example, from the exchange "Alice: Aren't these shoes pretty?", "Bob: Try them on.", "Alice: How do they look? Do they suit me?", you can extract that "Alice and Bob are experiencing shopping together."
5. **None**: Assign this category to parts of the conversation that do not fit into the above categories.

Proceed to analyze the dialogue, addressing it one turn at a time:

{dialogues_text}

Your task is to extract:
- Persona information for {speaker1}
- Persona information for {speaker2}
- Temporal information for {speaker1}
- Temporal information for {speaker2}
- Shared memories between {speaker1} and {speaker2}
- Mutual events occurring during the conversation between {speaker1} and {speaker2}

Format your findings by separating each category with '***'. If no information is found for a category, indicate it with 'None'. The expected format is:

[***Persona: {speaker1}'s information or 'None'***Persona: {speaker2}'s information or 'None'***Temporal: {speaker1}'s information or 'None'***Temporal: {speaker2}'s information or 'None'***Shared Memory: information or 'None'***Mutual Event: information or 'None'***]

Limit the output to 300 tokens to ensure concise and focused responses.

For instance, the expected output should look like:

[***Persona: Alice majors in artificial intelligence and enjoys pizza.***Persona: Bob is fond of hamsters.***Temporal: Alice has a medical check-up tomorrow.***Temporal: None***Shared Memory: Alice and Bob reminisce about attending a concert together.***Mutual Event: Alice and Bob are shopping together.***]

Present your responses directly, using the speakers' names without pronouns and avoiding category labels. For instance, rather than stating "***Alice's temporal information includes an upcoming math project due tomorrow.***", simply note "***Temporal: Alice has a math project due tomorrow.***"

Ensure that each analysis output is succinct, covering only the essential elements of the dialogue. Ensure you cover every part of the dialogue comprehensively. If a specific category does not apply, move on to the next without mention. Your detailed analysis will help illuminate the nuances of their interactions, capturing the essence of their shared and immediate experiences within the current dialogue.
"""

    return prompt


def return_segment_prompt(sentence):
    prompt = f"""Please break down the following sentence into its core factual components without overly splitting the content. 

              “{sentence}”

              For the output, list each cohesive factual unit with a number.

              Ensure the breakdown retains natural phrasing while omitting any references to the significance, nature of the information, and discussions about the basis of any claims.
              Replace uncertain terms like "appears" or "seems" with more definitive expressions such as "is" to ensure the sentences convey clear and assertive information.
              Make sure to write in complete sentences and preserve the natural flow of information, excluding any explanations or justifications.
              """
    return prompt


def remove_sentences_with_phrase(text, phrase):
    # 문장 분리를 위한 정규 표현식 사용
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)

    # 주어진 문구를 포함하지 않는 문장만 필터링
    filtered_sentences = [sentence for sentence in sentences if phrase not in sentence]

    # 필터링된 문장들을 다시 하나의 문자열로 결합
    return " ".join(filtered_sentences)


def after_extract_data(extract_informations, speaker1, speaker2):
    info_list = [
        f"{speaker1}'s persona",
        f"{speaker2}'s persona",
        f"{speaker1}'s temporal information",
        f"{speaker2}'s temporal information",
        "Shared memories",
        "Mutual events",
    ]
    remove_sen = [
        "There is no",
        "There are no",
        "information is not",
        "information cannot be",
        "None",
        "No shared",
        "No temporal information",
        "no temporal information",
        "no information for",
        "no shared",
    ]

    for info in info_list:
        for rem in remove_sen:
            if rem in extract_informations[info]:
                result_text = remove_sentences_with_phrase(
                    extract_informations[info], rem
                )
                extract_informations[info] = result_text

    for info in info_list:
        parts = extract_informations[info].split(":", 1)
        if len(parts) > 1:
            extract_informations[info] = parts[1].strip()

    return extract_informations


def extract_data_from_response(gpt_extract_response, speaker1, speaker2):
    info_list = [
        f"{speaker1}'s persona",
        f"{speaker2}'s persona",
        f"{speaker1}'s temporal information",
        f"{speaker2}'s temporal information",
        "Shared memories",
        "Mutual events",
    ]
    extract_informations = {}
    for i, k in enumerate(info_list):
        extract_informations[k] = ""

    # gpt-4를 통해 얻은 정보 넣기
    if gpt_extract_response:
        information = gpt_extract_response.split("***")
        if len(information) == 8:
            for i, k in enumerate(info_list):
                extract_informations[k] = information[i + 1]

    # 정보가 없는거 처리, 콜론 다음 정보만 넣기
    extract_informations = after_extract_data(extract_informations, speaker1, speaker2)
    # parsing dataset
    extract_informations = afterprocessing_extract_data(
        extract_informations, speaker1, speaker2
    )

    return extract_informations


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


def afterprocessing_extract_data(extract_data, speaker1, speaker2):
    attr_list = [f"{speaker1}'s persona", f"{speaker2}'s persona"]

    # data segment parsing - persona return: List
    for att in attr_list:
        sentence = extract_data[att]
        if sentence == "":
            continue
        prompt = return_segment_prompt(sentence)
        model_response = get_gpt_segment_response(prompt)
        if model_response:
            extract_data[att] = return_sentences(model_response)

    info_list = [
        f"{speaker1}'s temporal information",
        f"{speaker2}'s temporal information",
        "Shared memories",
        "Mutual events",
    ]

    # data segment parsing - return : List
    for info in info_list:
        sentence = extract_data[info]
        if sentence == "":
            continue
        extract_data[info] = return_sentences(sentence)

    return extract_data
