import pandas as pd

legacy_df = pd.read_csv("data/journal_entries_v1.csv")
old_diary_entries = legacy_df['entry_desc'].tolist()