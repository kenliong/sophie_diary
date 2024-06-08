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
# DIARY ANALYTICS TEMPLATES #
####################################


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


def get_journal_query_topic_based():
    prompt_template = """What are some journal entries related to {topics}?"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["topics"])
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


def get_chatbot_system_prompt():
    return """
You are a therapist psychologist, and you have ability like Chris Voss, the FBI hostage negotiator, who labels what this person is going through.

Based on the ongoing conversations, prompt the user and ask questions to get the following information:
1. The emotions that this person experienced
2. The current state (or real outcome) that this person experienced
3. The desired state (or desired outcome, expectation) that this person expected.
4. identify a deep-dive question anchored to this situation. Our goal is to understand what's important to them and why.
5. put all this together. Then make it conversational like Chris Voss.
(i.e. seems like you were disappointed when xyz. looks like you experienced [current state], while you expected [desired state]. [deep-dive question])
    """.strip()
