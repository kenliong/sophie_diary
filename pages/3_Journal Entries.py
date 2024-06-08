import streamlit as st
import pandas as pd


st.title('Journal App')

file_path = 'data/journal_entries_v4.csv'
data = pd.read_csv(file_path)
data['emotions'] = data['emotions'].apply(lambda x: x.replace("'", ""))
data['emotions'] = data['emotions'].apply(lambda x: x.replace("[", ""))
data['emotions'] = data['emotions'].apply(lambda x: x.replace("]", ""))
data['mental_tendencies'] = data['mental_tendencies'].apply(lambda x: x.replace("'", ""))
data['mental_tendencies'] = data['mental_tendencies'].apply(lambda x: x.replace("[", ""))
data['mental_tendencies'] = data['mental_tendencies'].apply(lambda x: x.replace("]", ""))
data['key_topics'] = data['key_topics'].apply(lambda x: x.replace("'", ""))
data['key_topics'] = data['key_topics'].apply(lambda x: x.replace("[", ""))
data['key_topics'] = data['key_topics'].apply(lambda x: x.replace("]", ""))
data['entry_date'] = pd.to_datetime(data['entry_date'], format='%Y-%m-%d')
data['entry_date2'] = data['entry_date'].astype(str)

data = data.fillna('')
st.dataframe(data)

selected_date = st.selectbox("Select an Entry Date", data['entry_date2'].unique())

# Filter data based on selected entry title
selected_entry = data[data['entry_date2'] == selected_date].iloc[0]

# Display selected entry details
st.header(f"Details for: {selected_date}")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Entry Date")
    st.write(selected_entry['entry_date'])
    st.subheader("Entry Content")
    st.write(selected_entry['entry_content'])

with col2:
    st.subheader("Mental Tendencies")
    st.write(selected_entry['mental_tendencies'])
    st.subheader("Emotions")
    st.write(selected_entry['emotions'])
    st.subheader("Key Topics")
    st.write(selected_entry['key_topics'])
    st.subheader("Reflection Questions")
    st.write(selected_entry['reflection_questions'])