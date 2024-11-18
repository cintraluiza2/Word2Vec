import spacy
import requests
from gensim.models import Word2Vec
import re

# Carregar o modelo spaCy 
nlp = spacy.load("en_core_web_sm")

# Baixar um arquivo de texto da internet
url = "https://pt.wikipedia.org/wiki/Immanuel_Kant"  # Exemplo de link para um livro público (em inglês)
response = requests.get(url)
text = response.text  # Obter o conteúdo do arquivo como string

# Verificar se o download foi bem-sucedido
if response.status_code == 200:
    print(f"Arquivo baixado com sucesso! Número de caracteres: {len(text)}")
else:
    print("Erro ao baixar o arquivo.")

def clean_text(text):
    # Remove HTML tags, links, and unnecessary symbols
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    return text    

# Processar o texto em frases (usando o spaCy para tokenizar)
corpus = text.split("\n")  # Dividindo o texto em linhas
tokenized_corpus = [ [token.text.lower() for token in nlp(sentence)] for sentence in corpus if sentence.strip()]

# Treinando o modelo Word2Vec
model = Word2Vec(sentences=tokenized_corpus, vector_size=500, window=20, min_count=5, workers=4)

# Salvando o modelo
model.save("word2vec_model_from_internet")

# Testando o modelo: Encontrar palavras semelhantes a "itself"
similar_words = model.wv.most_similar("kant", topn=10)
print(f"Palavras mais semelhantes a 'kant': {similar_words}")

# Encontrar similaridade entre duas palavras
similarity = model.wv.similarity('kant', 'filosofia')
print(f"Similaridade entre 'kant' e 'immanuel': {similarity}")

vector_kant = model.wv['kant']
vector_programming = model.wv['filosofia']
similarity = model.wv.cosine_similarities(vector_kant, [vector_programming])
print(f"Similaridade de 'kant' e 'filosofia': {similarity}")

# Obter o vetor para uma palavra
vector_kant = model.wv['kant']
print(f"Vetor de 'kant': {vector_kant}")
