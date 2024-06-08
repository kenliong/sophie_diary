from langchain.prompts import PromptTemplate

####################################
# Chat to Journal Prompt Templates #
####################################


##################################
# Journal Query Prompt Templates #
##################################

def get_journal_query_topic_based(topics):
    prompt_template = f"""What are some journal entries related to {topics}?"""
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