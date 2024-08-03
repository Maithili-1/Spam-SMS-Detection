#!/usr/bin/env python
# coding: utf-8



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

def load_data(file_path):
    data = pd.read_csv(file_path, encoding='latin-1')
    data = data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
    data = data.rename(columns={"v1": "Label", "v2": "Text"})
    return data

def preprocess_data(data):
    X = data['Text']
    y = data['Label']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    classifier = MultinomialNB()
    classifier.fit(X_train_tfidf, y_train)
    return classifier, tfidf_vectorizer

def evaluate_model(classifier, tfidf_vectorizer, X_test, y_test):
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Legitimate SMS', 'Spam SMS'])
    return accuracy, report

def display_progress():
    progress_bar = tqdm(total=100, position=0, leave=True)
    for i in range(10, 101, 10):
        progress_bar.update(10)
        progress_bar.set_description(f'Progress: {i}%')
    progress_bar.close()

def main():
    data = load_data("spam.csv")
    X_train, X_test, y_train, y_test = preprocess_data(data)
    classifier, tfidf_vectorizer = train_model(X_train, y_train)
    accuracy, report = evaluate_model(classifier, tfidf_vectorizer, X_test, y_test)
    
    display_progress()
    
    print(f'Accuracy: {accuracy:.2f}')
    print('Classification Report:')
    print(report)

if __name__ == "__main__":
    main()





