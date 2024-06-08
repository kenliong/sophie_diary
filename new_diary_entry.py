from typing import Optional, List
from pydantic import BaseModel, Field
from utils.llm_utils import *


class DeepDiveConversationLabels(BaseModel):
    current_state: Optional[str] = Field(
        default="",
        description="Label the current state (or real outcome) that this person experienced. If this information is not found, output 'None'."
    )
    desired_state: Optional[str] = Field(
        default="",
        description="Label the desired state (or desired outcome, expectation) that this person expected. If this information is not found, output 'None'"
    )
    

def extract_info_from_conversation(chat_history):
    parser = PydanticOutputParser(pydantic_object=DeepDiveConversationLabels)
     
    prompt = f'''
The following are thoughts from a user, extract the following information.
{parser.get_format_instructions()}
```
{chat_history}
```
    '''.strip()
    
    model = get_llm_instance()

    output = get_completion(model,prompt)
    parsed_output = parser.parse(output)
    
    return parsed_output

def check_conversation_labels(conversation_labels):

    if not conversation_labels.current_state or conversation_labels.current_state == 'None':
        return False
    if not conversation_labels.desired_state or conversation_labels.desired_state == 'None':
        return False

    return True

class DiaryEntrySummary(BaseModel):
    entry_title: str = Field(
        description="Summarize this person's desired state with a goal of understanding this person's value. Turn this person's value into the title. Write this in first person perspective."
    )
    entry_summary: str = Field(
        description="A 1 to 2 paragraph summary of the user's experience based on the conversation history between the user and an AI model. Write this in first person perspective."
    )

def summarize_new_entry(chat_model):
    chat_history = ''

    for msg in chat_model.history[2:]:
        chat_history += f"{msg.role}: {msg.parts[0].text} \n\n"

    parser = PydanticOutputParser(pydantic_object=DiaryEntrySummary)
     
    prompt = f'''
The following are thoughts from a user, extract the following information.
{parser.get_format_instructions()}
```
{chat_history}
```
    '''.strip()
    
    model = get_llm_instance()

    output = get_completion(model,prompt)
    parsed_output = parser.parse(output)
    
    print(parsed_output)
    
    return parsed_output
    