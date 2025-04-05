import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load the dataset
df = pd.read_csv('spam.csv')

# Preprocess the data
X = df['text']
y = df['label_num']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the text data into numerical data using CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Save the model and vectorizer to disk using joblib
joblib.dump(model, 'spam_detector_model.pkl') 
joblib.dump(vectorizer, 'vectorizer.pkl')    

print("Model and vectorizer saved successfully!")
