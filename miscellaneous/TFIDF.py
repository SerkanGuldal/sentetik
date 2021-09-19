from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

dataname = 'tr_hepsi_etiketli.csv' # Filename needs to be updated!!!!
df = pd.read_csv('twitter/' + dataname)


print(df)
print(df.shape)



vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df)

print(vectorizer.get_feature_names())
print(X.shape)