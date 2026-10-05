import pandas as pd

df = pd.read_csv(
    "./SMSSpamCollection.csv", sep="\t", header=None, names=["label", "Text"]
)
print(df)