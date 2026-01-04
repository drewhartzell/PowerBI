import pandas as pd

# Load TSV.GZ
basics = pd.read_csv(
    r"C:\Users\andre\Downloads\title.basics.tsv.gz",
    sep="\t",
    low_memory=False
)

episodes = pd.read_csv(
    r"C:\Users\andre\Downloads\title.episode.tsv.gz",
    sep="\t",
    low_memory=False
)

ratings = pd.read_csv(
    r"C:\Users\andre\Downloads\title.ratings.tsv.gz",
    sep="\t",
    low_memory=False
)

# Save as CSV
basics.to_csv(r"C:\Users\andre\Downloads\title.basics.csv", index=False)
episodes.to_csv(r"C:\Users\andre\Downloads\title.episode.csv", index=False)
ratings.to_csv(r"C:\Users\andre\Downloads\title.ratings.csv", index=False)

print("CSV files created successfully")
