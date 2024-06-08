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