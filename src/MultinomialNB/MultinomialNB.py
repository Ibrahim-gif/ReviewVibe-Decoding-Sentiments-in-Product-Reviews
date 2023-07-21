import pandas as pd
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


def read_csv(filename):
    temp = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            temp.append(' '.join(row))
    return temp

def train_predict(X_train, X_test, y_train, y_test, X_val, y_val, i, j):
    # Vectorize the text data using CountVectorizer
    vectorizer = CountVectorizer(ngram_range=(i, j))
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    X_val_vectorized = vectorizer.transform(X_val)

    alpha_list = [0.1, 0.5, 1.0, 2.0]
    best_accuracy = 0
    best_alpha = 0

    for alpha in alpha_list:
        classifier = MultinomialNB(alpha=alpha)
        # Train the Multinomial Naïve Bayes classifier
        classifier.fit(X_train_vectorized, y_train)

        # Predict the labels for the test set
        y_pred = classifier.predict(X_val_vectorized)

        # Calculate and print the classification accuracy
        accuracy = metrics.accuracy_score(y_val, y_pred)
        if (accuracy > best_accuracy):
            best_accuracy = accuracy
            best_alpha = alpha

    best_model = MultinomialNB(alpha=best_alpha)
    best_model.fit(X_train_vectorized, y_train)
    y_pred = best_model.predict(X_test_vectorized)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("The best alpha value is: ", best_alpha)
    return accuracy

def main():
    """Implement your assignment solution here"""

    X_train = read_csv('./data/train_val_test/train.csv')
    X_test = read_csv('./data/train_val_test/test.csv')
    y_train = read_csv('./data/train_val_test/train.label.csv')
    y_test = read_csv('./data/train_val_test/test.label.csv')
    X_val = read_csv('./data/train_val_test/val.csv')
    y_val = read_csv('./data/train_val_test/val.label.csv')

    unigram_accuracy = train_predict(X_train, X_test, y_train, y_test, X_val, y_val, 1, 1)
    bigram_accuracy = train_predict(X_train, X_test, y_train, y_test, X_val, y_val, 2, 2)
    uni_bi_gram_accuracy = train_predict(X_train, X_test, y_train, y_test, X_val, y_val, 1, 2)

    print("The unigram accuracy with stop words is :", unigram_accuracy)
    print("The bigram accuracy with stop words is :", bigram_accuracy)
    print("The uni + bi accuracy with stop words is :", uni_bi_gram_accuracy)

    #without stop words
    X_train = read_csv('./data/train_val_test/train_ns.csv')
    X_test = read_csv('./data/train_val_test/test_ns.csv')
    # y_train = read_csv('./data/train_val_test/train_ns.label.csv')
    # y_test = read_csv('./data/train_val_test/test_ns.label.csv')

    unigram_accuracy = train_predict(X_train, X_test, y_train, y_test, X_val, y_val, 1, 1)
    bigram_accuracy = train_predict(X_train, X_test, y_train, y_test, X_val, y_val, 2, 2)
    uni_bi_gram_accuracy = train_predict(X_train, X_test, y_train, y_test, X_val, y_val, 1, 2)

    print("The unigram accuracy without stop words is :", unigram_accuracy)
    print("The bigram accuracy without stop words is :", bigram_accuracy)
    print("The uni + bi accuracy without stop words is :", uni_bi_gram_accuracy)


if __name__ == "__main__":
    main()
