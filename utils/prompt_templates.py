from langchain.prompts import PromptTemplate

'''
READ THIS BEFORE ADDING PROMPTS
Specifications:
Each function should not require any inputs
The output it a PromptTemplate Objects
Inside each function define a string

e.g.
def get_concerns_prompt():
    prompt = """These are some of my concerns today {concerns_list}"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["topics"])
    return prompt

How the PromptTemplace Object is used:
prompt = get_prompt()
chain = LLMChain(llm=model, prompt=prompt)
result = chain.run({"context": context})
'''
####################################
# Chat to Metadata Prompt Templates #
####################################

def generate_reflection_questions_template():
    prompt_template = """
    This is the diary entry: {entry_content}.
    Recommend me 3 questions to reflect upon to improve this situation. 
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["entry_content"])
    return prompt

def generate_mental_tendencies_template():
    prompt_template = """
    Based on the content from this diary entry, can you tag it with the relevant mental tendencies?
    This is the diary entry: {entry_content}.
    This is the list of mental tendencies that you can tag from: {mental_tendencies}
    """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["entry_content", "mental_tendencies"]
    )
    return prompt


def generate_emotions_template():
    prompt_template = """
    Based on the content from this diary entry, can you tag it with the relevant emotions?
    This is the diary entry: {entry_content}.
    This is the list of emotions that you can tag from: {emotions}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["entry_content", "emotions"])
    return prompt


def generate_key_topics_template():
    prompt_template = """
    Based on the content from this diary entry, can you tag it with the relevant key topics?
    This is the diary entry: {entry_content}.
    This is the list of key topics that you can tag from: {key_topics}
    """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["entry_content", "key_topics"]
    )
    return prompt


##################################
# Journal Query Prompt Templates #
##################################

# def prompt_on_docs():
#     prompt_template = """
#         Here is some context of past journal entries {docs}
#         Here is my long term development plan {ltdp}
#         How do you feel about this?
#     """
#     prompt = PromptTemplate(template=prompt_template, input_variables=["docs"])
#     return prompt

def get_chat_starting_question():
    prompt_template = """
        We talked about the following in yesterday's journal: {ytd_chat}
        Help me come up with some ideas of what I should talk about today. Phrase it as a question.
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["ytd_chat"])
    return prompt

def get_topics_from_user_chat():
    prompt_template = """
        User has said this:{user_chat}
        What are some of the keyworks, topics, nouns mentioned in this user response?
        Provide a simple list of single words. No preamble.
        """
    prompt = PromptTemplate(template=prompt_template, input_variables=["user_chat"])
    return prompt

def get_journal_query_topic_based():
    prompt_template = """{user_chat} {topics}"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["user_chat", "topics"])
    return prompt


##################################
# Reply to User Prompt Templates #
##################################


def get_question_generation_template():
    prompt_template = """
    You are a pen pal for a friend who has been struggling with their mental health.
    Based on the following past diary entries, generate 5 relevant questions to prompt your friend to start journalling.
    Past Entries: {context}

    Questions:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    return prompt


##################################
# Chatbot system prompts         #
##################################


def get_chatbot_system_prompt(additional_info = '- The emotions that this person experienced', sahha_insights = '', similar_issues = ''):
    sahha_insights_output = ''
    if sahha_insights:
        sahha_insights_output = f'''
The user's biometric insights are as follows:
```
{sahha_insights}
```
        '''.strip()

    similar_issues_str = ''
    if similar_issues:
        similar_issues_str = f'''
The user had previously discussed these issues, you may reference these issues in your discussion with the user:
```
{similar_issues}
```
        '''.strip()
    
    prompt = f"""
{sahha_insights_output}
{similar_issues_str}

Based on the ongoing conversations, prompt the user and ask questions to get the following information. Keep your question friendly, simple and concise:
{additional_info}
    """.strip()

    return prompt