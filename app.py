import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# =====================================
# 1. Load Dataset
# =====================================

df = pd.read_csv("college_faqs.csv")

documents = df["question"] + " " + df["answer"]

# =====================================
# 2. Create Embeddings
# =====================================

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

doc_embeddings = embedding_model.encode(documents.tolist())

doc_embeddings = np.array(doc_embeddings).astype("float32")

# =====================================
# 3. Store in FAISS
# =====================================

dimension = doc_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(doc_embeddings)

# =====================================
# 4. Retrieval Function
# =====================================

def retrieve(query, top_k=1):

    query_embedding = embedding_model.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for i in indices[0]:
        results.append(df.iloc[i]["answer"])

    return results

# =====================================
# 5. Chatbot Function
# =====================================

def chatbot(query):

    answers = retrieve(query)

    if len(answers) > 0:
        return answers[0]

    return "Sorry, I don't know the answer."

# =====================================
# 6. Chat Loop
# =====================================

print("\n🎓 College FAQ Chatbot")
print("Type 'exit' to quit\n")

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    response = chatbot(user_input)

    print("Chatbot:", response)
    print()