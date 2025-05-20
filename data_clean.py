import pandas as pd

with open("dataset.csv", encoding='utf-8', errors='replace') as f:
    raw_df = pd.read_csv(f)

clean_df = raw_df.dropna(subset=["skill_id"])
clean_df = raw_df.dropna(subset=["skill_name"])
clean_df = clean_df[clean_df["skill_id"].astype(str).str.strip() != ""]
clean_df = clean_df[["user_id", "problem_id", "skill_id", "correct", "skill_name"]]

clean_df.to_csv("clean_dataset.csv")

print(clean_df)
