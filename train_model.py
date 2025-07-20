import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

messages = {
    'text': [
        "Congratulations, you've won a free iPhone!",
        "Click here to claim your prize",
        "Meeting scheduled at 10 AM tomorrow",
        "Please find the attached report",
        "Win money instantly, click the link",
        "Lunch at 1 PM?",
        "You are selected for a free vacation!",
        "Reminder: Project deadline is next Monday"
    ],
    'label': ['spam', 'spam', 'ham', 'ham', 'spam', 'ham', 'spam', 'ham']
}

df = pd.DataFrame(messages)
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=1)

cv = CountVectorizer()
X_train_vec = cv.fit_transform(X_train)

clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

joblib.dump(clf, "spam_model.pkl")
joblib.dump(cv, "vectorizer.pkl")
print("Model and vectorizer saved.")
