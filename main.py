from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


file_names = ['file1.txt','file2.txt', 'file3.txt']
documents = [open(file_name, 'r', encoding='utf-8').read() for file_name in file_names]

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(documents)

query = 'я ищу женские кроссовки'
query_vector = vectorizer.transform([query])

cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

sorted_doc_indices = cosine_similarities.argsort()[::-1]

for index in sorted_doc_indices:
    print(file_names[index])