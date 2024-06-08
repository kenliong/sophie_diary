import pandas as pd

old_diary_entries = pd.read_csv("data/journal_entries_v2.csv")

mental_tendencies = [
'High Expectations',
'External Validation',
'Reluctance to Share',
'Expectation of Reciprocity',
'Guilt-trapping',
'Balancing Correction and Encouragement',
'Demotivation from Initial Failures',
'Perfectionism',
'Tunnel Vision',
'Self-efficacy',
'Learned Helplessness',
'Overanalysis',
'Fear of Failure',
'Risk Aversion',
'Impostor Syndrome',
'Distractibility',
'Multitasking'
]

emotions = [
"Happiness",
"Sadness",
"Anger",
"Fear",
"Surprise",
"Disgust",
"Love",
"Anticipation",
"Trust",
"Jealousy",
"Envy",
"Guilt",
"Shame",
"Relief",
"Pride",
"Contempt",
"Frustration",
"Boredom",
"Hope",
"Confusion",
"Embarrassment"
]

key_topics = [
"Work and Career",
"Relationships",
"Health and Well-being",
"Personal Growth",
"Emotions and Feelings",
"Finances",
"Daily Life",
"Travel and Adventure",
"Spirituality and Beliefs",
"Technology and Innovation",
"Social Issues and Community",
"Creativity and Arts",
"Dreams and Aspirations"
]