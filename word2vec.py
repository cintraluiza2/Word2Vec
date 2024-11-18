import spacy
from gensim.models import Word2Vec

# Carregar o modelo spaCy (utilizando o modelo de português - pt_core_news_sm)
nlp = spacy.load("en_core_web_sm")

# Exemplo de um pequeno corpus de texto
corpus = [
    "I really wanted to die",
    "Sometimes life feels lonely",
    "My life is a shit",
    "I really enjoy being auto-mutilated",
    "I wish I could be more independent"
]

# Tokenizar o texto usando spaCy (dividir as frases em palavras)
tokenized_corpus = [ [token.text.lower() for token in nlp(sentence)] for sentence in corpus]

# Treinando o modelo Word2Vec
model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)

# Salvando o modelo
model.save("word2vec.model")

# Carregar o modelo (caso necessário)
model = Word2Vec.load("word2vec.model")

# Testando o modelo: Encontrar palavras semelhantes a "python"
similar_words = model.wv.most_similar("die", topn=3)
print(f"Palavras mais semelhantes a 'die': {similar_words}")

# Encontrar similaridade entre duas palavras
similarity = model.wv.similarity('die', 'lonely')
print(f"Similaridade entre 'die' e 'lonely': {similarity}")

# Obter o vetor para uma palavra
vector_die = model.wv['die']
print(f"Vetor de 'die': {vector_die}")
