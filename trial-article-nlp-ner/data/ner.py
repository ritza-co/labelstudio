import spacy
import pandas as pd

nlp = spacy.load("en_core_web_sm")


df = pd.read_csv("lines_clean.csv")
df = df[df['line_text'].str.contains("Easter ")]
print(df.shape)

texts = df['line_text']

for text in texts[:10]:
    doc = nlp(text)
    
    for token in doc:
        print(token.text, token.ent_type_)
    print("----------")



