def return_model_prompt(dia_text):
  prompt = f"""
Task: Generate the next response in a dialogue by focusing on the contextual cues detailed within parentheses in the dialogue history. Responses should be tailored according to the type of cue provided:

1. Memory-driven dialogues: If the cue within parentheses details specific character traits or background context, craft responses that reflect these memory-driven elements, ensuring character consistency and rich context.
2. Everyday language dialogues: If the cue within parentheses is labeled "Everyday Language," generate responses that are based on typical day-to-day interactions, free from specific personas or detailed context.

**Dialogue History**:
{dia_text}

"""
  return prompt

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