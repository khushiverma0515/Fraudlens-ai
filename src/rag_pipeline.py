import os

import faiss
from sentence_transformers import SentenceTransformer


# 1. KNOWLEDGE BASE LOCATION
KNOWLEDGE_BASE_FOLDER = "src/knowledge_base/fraud_guidelines"


# 2. LOAD KNOWLEDGE DOCUMENTS
documents = []

for filename in os.listdir(KNOWLEDGE_BASE_FOLDER):
    if filename.endswith(".txt"):
        file_path = os.path.join(KNOWLEDGE_BASE_FOLDER, filename)

        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        documents.append({
            "filename": filename,
            "text": text
        })


print("--- Knowledge Base Loaded ---")
print("Documents:", len(documents))

for document in documents:
    print("-", document["filename"])


# 3. LOAD EMBEDDING MODEL
print("\n--- Loading Embedding Model ---")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded!")


# 4. CREATE EMBEDDINGS
texts = [document["text"] for document in documents]

embeddings = model.encode(
    texts,
    convert_to_numpy=True
)


# 5. NORMALIZE EMBEDDINGS
faiss.normalize_L2(embeddings)


# 6. CREATE FAISS SEARCH INDEX
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print("\n--- FAISS Index Created ---")
print("Documents indexed:", index.ntotal)


# 7. RETRIEVE RELEVANT INFORMATION
def retrieve_context(query, k=2):
    """Retrieve relevant fraud guidance for a question."""

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    faiss.normalize_L2(query_embedding)

    number_of_results = min(k, index.ntotal)
    scores, results = index.search(
        query_embedding,
        number_of_results
    )

    retrieved_chunks = []

    for result_index in results[0]:
        if result_index == -1:
            continue

        document = documents[result_index]

        retrieved_chunks.append(
            f"Source: {document['filename']}\n{document['text']}"
        )

    return "\n\n".join(retrieved_chunks)


# 8. TEST THE RETRIEVAL FUNCTION
if __name__ == "__main__":
    query = "What transaction characteristics may indicate fraud?"
    context = retrieve_context(query)

    print("\n--- RAG Search ---")
    print("Query:", query)
    print("\n--- Retrieved Context ---")
    print(context[:2000])