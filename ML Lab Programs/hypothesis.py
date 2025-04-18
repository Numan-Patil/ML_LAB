import pandas as pd

df = pd.read_csv("C:/Users/numan/OneDrive/Desktop/ML Lab Programs/training_data.csv")

attributes = df.columns[:-1]
target_col = df.columns[-1]

hypothesis = ['Ø'] * len(attributes)

for _, row in df.iterrows():
    if row[target_col].lower() == 'yes':
        for i in range(len(attributes)):
            if hypothesis[i] == 'Ø':
                hypothesis[i] = row[i]
            elif hypothesis[i] != row[i]:
                hypothesis[i] = '?'

print("Final Hypothesis using Find-S algorithm:")
print(hypothesis)
