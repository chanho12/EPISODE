
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
	with open(file_path, 'r') as file:
		data = json.load(file)
	return data

def get_gpt_response(prompt):
    
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
            # {
            # "role": "user",
            # "content": [
            #     {
            #     "type": "text",
            #     "text": "Based on the provided conversation history, identify three responses that would appropriately continue the dialogue. Choose the response labeled \"everyday language\" only if no other options are suitable.\n\nConversation History:\nsidney : (sidney shows an assertive personality and expresses his intentions of giving advice to susan like an uncle) wait for me ill be right back its not my nature susie but ill talk to you like an uncle susan : (susan displays independence and self-assuredness in declining sidney's offer to act as an uncle) but i dont need an uncle sidney sidney : (sidney demonstrates a certain level of admiration toward susan , sidney plays a role of mediator between susan and her brother, and possibly between susan and j.j. , sidney has plans to meet j.j. in the near future , susan has a love message for j.j. about steve dallas that she wants sidney to deliver) no i mean because i admire you in fact more than admire you although thats neither here nor there susie dont sell your brother short talk this over with him i mean youll find him a real friend any message in case i see jj later\nCandidate Responses:\n0. susan observes sidneys tendency to be touchy 1. susan is in love with a man named steve 2. susan displays independence and selfassuredness in declining sidneys offer to act as an uncle 3. susan is experiencing romantic feelings for the first time with a man named steve dallas 4. susan is jjs sister 5. susan is upset with both sidney and steve 6. susan is contemplating her future including her decision on whether to join steve for a long tour and her plan to discuss her relationship with steve with her brother jj the next morning she is also exercising introspection reflecting on her feelings towards her familial relationships and her romantic interest 7. susan desires for sidney and steve to get along 8. susan hints at a lack of selfworth when calling herself a worthless rag 9. susan seeks validation for her own identity separate from her brother 10. susan is considering embarking on an eightmonth tour with steve 11. susan is in a state of high emotional distress 12. susan is possibly unaware of a family members illness 13. susan is helpless and somewhat disorganized 14. everyday language 15. susan mentions giving up steve and regrets the loneliness of her brother suggesting a matured sense of responsibility 16. susan had contact with mr dangelo and steve 17. susan made a visit to the hospital 18. susan is distraught and feels guilt and sorrow indicating an emotional and sensitive nature 19. susan feels constricted by her identity as her brothers sister 20. susan may be younger or less experienced in relationships\n\nSelect the Appropriate Responses:\nPlease specify the responses in the following format:\nnumber1 : candidate response text\nnumber2: candidate response text\nnumber3 : candidate response text"
            #     }
            # ]
            # },
            # {
            # "role": "assistant",
            # "content": [
            #     {
            #     "type": "text",
            #     "text": "1: susan is in love with a man named steve\n7: susan desires for sidney and steve to get along\n10: susan is considering embarking on an eight-month tour with steve"
            #     }
            # ]
            # },
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": "Based on the provided conversation history, identify up to three responses that would most appropriately continue the dialogue. Prioritize your selections based on their relevance and coherence with the previous exchange. Choose the response labeled \"everyday language\" only if no other suitable options are available.\nConversation History: \nMIKE : (Everyday Language) Were you outside? KYLE : (KYLE regularly runs) Yeah. I went for a run. MIKE : (MIKE is surprised) It's pretty early. You do this every morning? KYLE : (KYLE regularly drills moves in the morning , KYLE does this before school during his sports season , KYLE's practice routine is in season) In season, yeah. I like to get my run in early and then drill some moves for an hour before school. MIKE : (MIKE is surprised) Oh. Wow.\nCandidate Responses:\n0. KYLE is hesitant. 1. KYLE is naive about the concept of guardianship. 2. KYLE believes he is pretty good at wrestling. 3. KYLE indicates he's self-reliant. 4. everyday language 5. KYLE visited his grandfather recently. 6. KYLE is distressed. 7. KYLE regularly runs and drills moves in the morning. 8. KYLE is naive about the situation of his grandfather. 9. KYLE wants to avoid going with someone. 10. KYLE shows signs of being a rebel. 11. KYLE smokes. 12. MIKE alludes to a shared memory of a recent victory that was significant to their team. 13. KYLE is involved in coaching with Stemler. 14. KYLE refuses financial assistance. 15. KYLE does this before school during his sports season. 16. KYLE possibly demonstrates leadership capabilities. 17. KYLE used to wrestle but stopped a couple of years ago. 18. KYLE feels in control. 19. KYLE last wrestled a couple of years ago. 20. KYLE decided not to go home and broke into his grandfather's house. 21. KYLE is good at something.\n\n\nInstructions for Selection: In your response, prioritize and indicate which candidate responses are most suitable for continuing the dialogue. Ensure your selections reflect the continuity and context of the conversation.\nSelect the Appropriate Responses:\nPlease specify the responses in the following format:\nnumber1 : candidate response text\nnumber2: candidate response text\nnumber3 : candidate response text"
                }
            ]
            },
            {
            "role": "assistant",
            "content": [
                {
                "type": "text",
                "text": "3: KYLE indicates he's self-reliant.\n7: KYLE regularly runs and drills moves in the morning.\n16: KYLE possibly demonstrates leadership capabilities."
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

def return_prompt(dialogues_text, candidates):
    
    prompt = f""" 
Based on the provided conversation history, identify up to three responses that would most appropriately continue the dialogue. Prioritize your selections based on their relevance and coherence with the previous exchange. Choose the response labeled "everyday language" only if no other suitable options are available.
Conversation History:
{dialogues_text}

Candidate Responses:
{candidates}

Instructions for Selection: In your response, prioritize and indicate which candidate responses are most suitable for continuing the dialogue. Ensure your selections reflect the continuity and context of the conversation.

Select the Appropriate Responses:
Please specify the responses in the following format:
number1 : candidate response text
number2 : candidate response text
number3 : candidate response text
"""
    return prompt


def memory_update(data):
	count = 0
	for key, value in data.items():
		count += 1
		if count > 5:
			break
		speaker1, speaker2 = tuple(key.replace("(", "").replace(")", "").replace("'", "").split(", "))
		conversations = value['dialogue']

		for idx, session in enumerate(conversations):
			if idx == 0:
				continue
			previous_memory = ""
			current_memory = ""
			attribute = [f"{speaker1}'s persona", f"{speaker1}'s temporary event", f"{speaker2}'s persona", f"{speaker2}'s temporary event"]

			for att in attribute:
				previous_att = value['dialogue'][idx - 1][att]
				current_att = session[att]
				if len(previous_att) > 0:
					previous_memory += f"- {att}: \n"
				for pre in previous_att:
					previous_memory += pre + '\n'
				if len(current_att) > 0:
					current_memory += f"- {att} : \n"
				for cur in current_att:
					current_memory += cur + '\n'

			print(previous_memory)
			print('-'*100)
			print(current_memory)
			print('^'*100)



###main###
def main():
    file_name = "list_dataset.json"
    data = load_data(file_name)
    memory_update(data)


if __name__ == '__main__':
    main()