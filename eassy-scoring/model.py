from textblob import TextBlob
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
import pickle

def extractFeatures(essayText, promptText):
    words = essayText.split()
    grammarErrors = sum(1 for w in words if TextBlob(w).correct() != w)
    sentences = essayText.split(".")
    avgSentenceLength = len(words) / max(1, len(sentences))

    vectorizer = TfidfVectorizer().fit([essayText, promptText])
    essayVec = vectorizer.transform([essayText])
    promptVec = vectorizer.transform([promptText])
    similarity = (essayVec @ promptVec.T).toarray()[0][0]

    return [grammarErrors, len(words), avgSentenceLength, similarity]

# Dummy training dataset
trainData = [
    {"essay": "This essay is very well written. It contains strong arguments.", "prompt": "Write an essay about education.", "score": 9},
    {"essay": "This essay have many mistake. Grammar is bad.", "prompt": "Write an essay about education.", "score": 4},
    {"essay": "The essay is average. Some points are clear but others are weak.", "prompt": "Write an essay about education.", "score": 6},
    {"essay": "An excellent essay with good structure and grammar.", "prompt": "Write an essay about education.", "score": 10},
]

# Train model
features = []
labels = []
for row in trainData:
    features.append(extractFeatures(row["essay"], row["prompt"]))
    labels.append(row["score"])

model = LinearRegression()
model.fit(features, labels)

# Save trained model
with open("essayModel.pkl", "wb") as f:
    pickle.dump(model, f)
