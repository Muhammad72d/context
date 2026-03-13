from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

document = '''
Artificial Intelligence (AI) is a field that focuses on creating machines capable of intelligent behavior.
Machine Learning (ML) is a subset of AI that enables machines to learn from data.
Deep Learning (DL) uses neural networks with multiple layers for complex pattern recognition.
Applications of AI include natural language processing, computer vision, robotics, healthcare, finance, and recommendation systems.
Ethical AI ensures fairness, accountability, and transparency in AI deployments.
AI continues to evolve rapidly, impacting numerous industries and research domains.
Robots are used in manufacturing, healthcare, exploration, and domestic assistance.
Natural Language Processing enables machines to read, understand, and generate human language.
Computer Vision allows machines to interpret and make decisions based on visual data.
The future of AI is expected to bring advancements in general AI and human-AI collaboration.
'''

print(document)


def chunk_text(text, chunk_size=30):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

chunks = chunk_text(document, chunk_size=30)
print('Number of chunks:', len(chunks))
chunks


embedder = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = embedder.encode(chunks)
print('Embedding shape:', chunk_embeddings.shape)


query = "What are the applications of AI?"

query_embedding = embedder.encode([query])

similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]

top_indices = np.argsort(similarities)[-2:][::-1]
relevant_chunks = [chunks[i] for i in top_indices]

relevant_chunks


if len(document.split()) > 100:
    summary = ' '.join(document.split()[:30]) + '...'
    print('Document summary:', summary)

context_text = ' '.join(relevant_chunks)

prompt = f"Context: {context_text}\n\nQuestion: {query}\nAnswer:"
print(prompt)
