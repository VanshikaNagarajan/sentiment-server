import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

svm_model = joblib.load('model/model.joblib')


def tokenize(string):
    tfidf_vectorizer = TfidfVectorizer(max_features=4000)
    v = tfidf_vectorizer.fit_transform(string)
    print(v)
    return v


def predict(tokens):
    result = svm_model.predict(tokens)
    print(result)
    return result


def handle_request(string):
    if True:
        return string[::-1]
    tokens = tokenize(string)
    return predict(tokens)
