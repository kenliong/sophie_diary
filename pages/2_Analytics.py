import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
from utils.llm_utils import get_sahha_insights


sahha_prompt, well_being_score = get_sahha_insights(1,1)

# Load the data
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

# Convert entry_date to datetime
data['entry_date'] = pd.to_datetime(data['entry_date'], format='%Y-%m-%d')

# Page title
st.title("Journal Entries Analytics")

# Display number of journal entries and latest journal entry date
num_entries = len(data)
latest_entry_date = data['entry_date'].max().strftime('%Y-%m-%d')
col1, col2, col3 = st.columns(3)
col1.metric("Well Being Score (Powered by Sahha)", well_being_score)
col2.metric("Number of Journal Entries", num_entries)
col3.metric("Latest Journal Entry Date", latest_entry_date)

# Most common emotions
st.header("Most Common Emotions")
emotions = data['emotions'].str.split(', ').explode()
emotions_count = emotions.value_counts()
fig_emotions_count = px.bar(emotions_count, labels={'index': 'Emotions', 'value': 'Count'})
st.plotly_chart(fig_emotions_count)

# Word cloud of key topics
# st.header("Word Cloud of Key Topics")
# key_topics = ' '.join(data['key_topics'].dropna().tolist())
# wordcloud = WordCloud(width=800, height=400, background_color='white').generate(key_topics)
# st.image(wordcloud.to_array(), use_column_width=True)

# Word cloud of mental tendencies
# st.header("Word Cloud of Mental Tendencies")
# mental_tendencies = ' '.join(data['mental_tendencies'].dropna().tolist())
# wordcloud = WordCloud(width=800, height=400, background_color='white').generate(mental_tendencies)
# st.image(wordcloud.to_array(), use_column_width=True)

# Heatmap of emotions over time
st.header("Heatmap of Emotions Over Time")
emotions_over_time = data[['entry_date', 'emotions']].copy()
emotions_over_time = emotions_over_time.set_index('entry_date').emotions.str.get_dummies(sep=', ').resample('M').sum()
emotions_over_time = emotions_over_time.reset_index().melt(id_vars='entry_date', var_name='emotions', value_name='Count')
fig_heatmap = px.density_heatmap(emotions_over_time, x='entry_date', y='emotions', z='Count',
                                 color_continuous_scale='Viridis', labels={'entry_date': 'Date', 'Count': 'Count'})
st.plotly_chart(fig_heatmap)

# Bubble chart of key topics
st.header("Bubble Chart of Key Topics")
key_topics_list = data['key_topics'].str.split(', ').explode().value_counts().reset_index()
key_topics_list.columns = ['Key Topic', 'Frequency']
fig_bubble = px.scatter(key_topics_list, x='Key Topic', y='Frequency', size='Frequency', hover_name='Key Topic',
                        size_max=60, labels={'Key Topic': 'Key Topics', 'Frequency': 'Frequency'})
st.plotly_chart(fig_bubble)

# Sentiment analysis over time
st.header("Sentiment Analysis Over Time")
data['sentiment'] = data['entry_content'].apply(lambda x: TextBlob(x).sentiment.polarity)
sentiment_over_time = data[['entry_date', 'sentiment']].set_index('entry_date').resample('M').mean().reset_index()
fig_sentiment = px.line(sentiment_over_time, x='entry_date', y='sentiment', labels={'entry_date': 'Date', 'sentiment': 'Average Sentiment'})
st.plotly_chart(fig_sentiment)