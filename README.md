# Sophie's Diary

## Problem statement

Sophie's Diary is an AI-powered app designed to enhance emotional and mental well-being. By aggregating data from journal entries and health metrics such as sleep and activity, the app provides personalized feedback, identifies areas for improvement, and suggests small, actionable steps.

For users who do not journal, the app encourages reflection through short prompts. This helps users understand their emotional states, offering insights into their behavior over time and guiding them towards their desired well-being goals.

## Key value proposition

Sophie's Diary offers personalized insights that are refined over time. Leveraging AI and LLM technology, the app provides intelligent, timely and cost-effective support.

## Features

1. **Journal Entry System**: Users begin by interacting with the app. They can either input their own journal entries or respond to short prompts provided by the app if they don't have anything specific to write about. This helps users express their emotional state and explore the reasons behind their feelings.

2. **Therapist LLM**: An AI-driven therapist engages with users, understanding their current and desired emotional states. The conversation continues until a sufficient amount of information has been gathered. During this process, the app provides suggestions on how users can improve their emotional well-being, especially if they are experiencing negative emotions.

3. **Data Storage and Analysis**: During the conversation, data is stored in a database to maintain a long-term history of the user's profile. The following data points are collected and analyzed:

   - Conversation Labels: Labels indicating the user's feelings.
   - Summary of each entry: Summaries of each journal entry based on the entire conversation.
   - Behavioral Tendencies: Analysis of the user's behavioral patterns, personal values, and areas for improvement.
   - Health Metrics: Integration of Sahha's health metrics, including activity levels and sleep patterns, to provide a holistic view of the user. These metrics are used to understand how physical health influences moods and actions.

4. **Live Dashboard Updates**: After each journal entry, a dashboard is updated with insights into the user's emotions and health scores, providing a comprehensive view of their emotional state over time. Key components include:
   - A bar chart showing the most common emotions.
   - A word cloud describing predominant feelings.
   - A heatmap displaying the user's emotions over time.
   - Flags for traits indicating lack of sleep or activity, based on Sahha's health definitions.

## Getting started

### Prerequisites

1. Refer to `requirements.txt` for dependencies to download. To download, run `pip install -r requirements.txt`
2. Create a `.env` file to store the `GOOGLE_API_KEY`. The Gemini model is used.

### User journey

1. Start the application with `streamlit run main.py`
2. Begin by talking about how you feel or what happened today.
3. Click 'Submit'
4. Interact with the Therapist LLM to discuss your feelings & desired emotional state
5. After completing your journal entry, view the updated insights on your dashboard alongside personal actionable steps that you can take to improve yor health.

### IMPORTANT NOTE:

This app is inspired by our teammate, Sophie. No mock data is used. All the data currently stored in the database are some of her past diary entries. Hence, if you notice that the AI-therapist has made some assumptions about your feelings or past experiences, it is due to the data that is currently stored. The LLM uses RAG techniques to search the database and chat with users, which will result in responses influenced by Sophie's past diaries.

We plan to make improvements and let other users try this app. In future iterations, each new user will have a clean database profile.

## Technical Design

![Sophies Diary Overview](https://github.com/kenliong/sophie_diary/assets/71979039/a4252a9b-9c88-42bc-af85-3bac8efffbaa)

![image](https://github.com/kenliong/sophie_diary/assets/52147112/18edb372-8b45-467b-8635-8e4d4c65fd80)

### Health Indicators Retrieval

Integration of the Sahha API is currently not possible. As a temporary solution, we have obtained a static JSON file for a given day from the Sahha team. We assume the user_id to be 4, and the data represents the user's health metrics for that day. This data includes physiological, mental, and activity signals that provide insights into the user's current state of health.

### Chat Loop

![image](https://github.com/kenliong/sophie_diary/assets/52147112/4530412d-3cfc-4aa5-af19-d0676fd2c156)

Chat loop does a bunch of things:

1. Initiating the conversation: The user starts a conversation with the AI therapist, talking about anything they have in mind or responding to a prompt if they don't have a specific topic.

2. Message evaluation: Each user message is evaluated to understand their emotional state and assess progress towards feeling better or more satisfied (desired state).

   - Desired state check: The AI therapist continuously evaluates whether the user feels better or has gotten what they wanted from the conversation. Once the user has reached their desired state, it proceeds to store the diary analytics into the relational database AND vector store (FAISS). This data will be used in subsequent steps for a similarity search to provide context for future conversations.
   - Else, the conversation continues.

3. Conversation is ongoing: A variety of data sources are used to construct a valid response by the Therapist LLM:

   - Sahha Data
   - Topic Extraction: The user message undergoes topic extraction, and the extracted topics are stored in the vector store. Similarity searches are performed to find related topics from past conversations.
   - The full chat history of the user

4. Personalized responses: The Response Constructor aggregates the data from various sources and makes a call to the LLM, optimizing prompts using LangChain chains. This involves RAG techniques on the vector database, which includes past diary entries, metadata, and Sahha API data. The constructed response is then sent back to the user, looping back to Step 2 for continuous evaluation.

### Real-time data analytics

Upon completion of the chat, chat data and metadata are processed into Diary Analytics that can be stored in the Vector Data Store and Relational Database Store (currently CSV). Using LangChain and advanced prompt techniques, we extract the following attributes from a single diary entry:

- Emotions
- Key Topics
- Mental Tendencies
- Reflection Questions

### Some engineering considerations:

1. UUID is used to uniquely tag each journal entry
2. Datetime data is collected at the moment a diary entry is completed and stored as part of the metadata, adding chronology and enabling potential analysis of user behavior over time.
3. Simple and intuitive design

### Long Term Storage

#### Vector Store

The vector store allows us to store previous and current journal entries and conversations for future reference. The LLM can follow up with the user based on past interactions, making each entry and conversation more personalized and relevant.

#### Relational Database Store

The relational database store enables efficient access and storage of diary analytics with high granularity for multiple users. Currently, due to time constraints, we are using a simple CSV approach. In the future, we aim to transition to a relational database for better scalability and data management.

### In Progress

#### Personality Summarization

In the future, we would like to encapsulate the user's personality traits so as to provide a more in-depth look into their personality. The LLM can also use it to consider different approaches to the conversation. Goal is to design a future you that is more resilient.

#### Future ideas

- Customizable goals: Allow users to set and track personal goals within the app, with the AI providing actionable steps, reminders, and potentially gamifying the experience.
- Community support: Create a feature for users to connect with a supportive community, sharing experiences and advice.
