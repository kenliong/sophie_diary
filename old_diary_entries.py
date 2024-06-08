import pandas as pd

legacy_df = pd.read_csv("data/journal_entries_v2.csv")
old_diary_entries = legacy_df['entry_desc'].tolist()

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