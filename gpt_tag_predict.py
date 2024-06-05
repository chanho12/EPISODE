from openai import OpenAI
import os


api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


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
    

def return_tag_prompt(dialogues_text, candidates):
  prompt = f"""
Based on the provided conversation history, identify three responses that would appropriately continue the dialogue. Choose the response labeled "everyday language" only if no other options are suitable.

Conversation History:
{dialogues_text}

Candidate Responses:
{candidates}

In your response, indicate which candidate responses are most suitable for continuing the dialogue. 
Ensure your selections reflect the continuity and context of the conversation.

Select the Appropriate Responses:
Please specify the responses in the following format:
number1 : candidate response text
number2 : candidate response text
number3 : candidate response text
"""
    
  return prompt


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
    gpt_candidate = ", ".join([
    f"{candidates[idx].rstrip('.')}" 
    for idx in gpt_choice
    if not (len(gpt_choice) >= 2 and candidates[idx] == 'Everyday Language')
    ])
    #gpt_candidate = candidates[gpt_choice[0]].rstrip('.')
  else:
    gpt_candidate = "Everyday Language"
  
  print(f"gpt predict: ", gpt_candidate)
  return gpt_candidate
