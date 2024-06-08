# Sophie's Diary
Sophie's Diary is designed to help users understand and improve their emotional and mental well-being. By facilitating journal entries and providing actionable steps, the app aids users in reaching their desired emotional states & offers insights into their behavior over time.

## Features
1. **Journal Entry System**: Users can start their journal entries by discussing their emotional state and why they're feeling this way.
2. **Therapist LLM**: An AI-driven therapist is instantiated to interact with the user, understanding their current and desired emotional states. The conversation continues until the user's desired state is reached.
3. **Data Storage and Analysis**: During the conversation, the following data points are stored:
    - Conversation Labels: Labels indicating the user's feelings.
    - Summary of Entry: Summaries of each journal entry for future reference.
    - Behavioral Tendencies: Analysis of the user's behavioral patterns, personal values, and areas for improvement.
4. **Dashboard Updates**: After each journal entry, a dashboard is updated with insights into the user's emotions and health scores, providing a comprehensive view of their emotional state over time. Key components include:
    - A bar chart showing the most common emotions.
    - A word cloud describing predominant feelings.
    - A heatmap displaying the user's emotions over time.
    - Flags for traits indicating lack of sleep or activity, based on Sahha's health definitions.

## Getting Started

### Prerequisites
Refer to `requirements.txt` for dependencies to download. To download, run `pip install -r requirements.txt`

### Usage
1. Start the application with `streamlit run main.py`
2. Begin by talking about how you feel or what happened today.
3. Click 'Submit'
4. Click 'Explore Further' to interact with the Therapist LLM and discuss your feelings & desired emotional state
5. After completing your journal entry, view the updated insights on your dashboard alongside personal actionable steps that you can take to improve yor health.

## Technical Design

![image](https://github.com/kenliong/sophie_diary/assets/52147112/9cf7e993-cb33-4959-8458-0c5a03bc3e4c)

### Health Indicators Retrieval

Integration of Sahha API is not possible now so as aligned, we have retrieved a static json file for a given day from the Sahha team. We assume user_id to be 4 and the output is from a given day. The output data contains physiological, mental and activity signals that can inform the user's current state of health. The

### Chat Loop

Chat Loop does a bunch of things:
1. Upon receiving user message, it checks if the message has completed
    - If completed (i.e. user has reached their desired state), it proceeds to store diary analytics into the relational database AND vector store, which can be picked up via similarity search for future conversations
    - Else, proceed to Step 2
2. Conversation is ongoing: A variety of data sources are used to construct a valid response by the Therapist LLM:
    - Sahha Data
    - Topic Extraction occurs from the user message, which is stored in Vector Store & similarity search is performed to find similar topics from past conversations
    - The original user message
3. Personal Conversation: Response Constructor collates the data sources before making a call to the LLM and returns a response (back to Step 1)


### Diary Analytics

Upon completion of the chat, chat data & metadata is processed into Diary Analytics that can be stored into the Vector Data Store (currently using FAISS) and Relational Database Store (currently CSV). Specific data that gets stored is:

- Emotions
- Key Topics
- Mental Tendencies
- Reflection Questions

### Long Term Storage

#### Vector Store

The vector store allows us to store previous and current journal entries & conversations for use in the future. The LLM can follow up with the user based on previous conversations, if relevant, making the entry & conversation more customized.

#### Relational Database Store

This relational data store would allow us to efficiently access & write in diary analytics with high granularity, for many users. Due to time constraints, we opted for a naive CSV approach. In the long run, we hope to use a relational database to store such details.

### In Progress

#### Personality Summariztion

In the future, we would like to encapsulate the user's personality traits so as to provide a more in-depth look into their personality. The LLM can also use it to consider different approaches to the conversation. Goal is to design a future you that is more resilient.
