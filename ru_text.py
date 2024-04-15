from nltk import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

LANGUAGE = 'russian'

def delete_stop_words(tokens):
    stop_words = set(stopwords.words(LANGUAGE))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens


def tokenize_and_stem(text):
    stemmer = SnowballStemmer(LANGUAGE)
    tokens = word_tokenize(text)
    filtered_tokens = delete_stop_words(tokens)
    return [stemmer.stem(token) for token in filtered_tokens]

def print_result(sorted_doc_indices):
    for index in sorted_doc_indices:
        print(file_names[index])

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


# Имена файлов
file_names = ['file1.txt', 'file2.txt', 'file3.txt']


# Чтение содержимого файлов
documents = [read_file(file_name) for file_name in file_names]

vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem)

# Обучение vectorizer на содержимом файлов
tfidf_matrix = vectorizer.fit_transform(documents)

query = 'я ищу кроссовки из синтетической кожи'
query_vector = vectorizer.transform([query])

cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

sorted_doc_indices = cosine_similarities.argsort()[::-1]

print_result(sorted_doc_indices)
