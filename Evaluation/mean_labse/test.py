from sentence_transformers import SentenceTransformer, util

sentences = ["I like this movie.", "I don't like this movie."] # 0.8435
sentences = ["I like this movie.", "I don't enjoy this film."] # 0.7891
sentences = ["Seen James?", "Could you tell me if you have seen James?"] # 0.7814
sentences = ["Sorry about that.", "I apologize for the inconvenience caused."] # 0.4476

model = SentenceTransformer('sentence-transformers/LaBSE')
embeddings = model.encode(sentences, convert_to_tensor=True)

cosine_scores = util.cos_sim(embeddings[0], embeddings[1])

print(embeddings)
