# EXPERIMENT 8
# RAG Data Indexing using FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# 1. Sample document
text = """
Climate change is a major environmental challenge. It causes rising
global temperatures and changes in weather patterns. Climate change
can lead to melting glaciers and rising sea levels.

Extreme weather events such as floods, droughts, storms and heatwaves
are becoming more common. These events can affect agriculture, water
resources and human health.

Coastal cities are especially vulnerable because rising sea levels can
cause flooding. Reducing greenhouse gas emissions and using renewable
energy can help reduce the effects of climate change.
"""


# 2. Split text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30
)

chunks = splitter.create_documents([text])

print("Number of chunks:", len(chunks))


# 3. Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# 4. Create FAISS vector store
db = FAISS.from_documents(chunks, embeddings)


# 5. Query
query = "What are the effects of climate change?"


# 6. Search top 2 chunks
results = db.similarity_search_with_relevance_scores(
    query, k=2
)


# 7. Display results
print("\nQuery:", query)
print("\nTop 2 chunks:")

for doc, score in results:
    print("\nScore:", round(score, 2))
    print("Chunk:", doc.page_content)