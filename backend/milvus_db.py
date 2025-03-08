from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
from sentence_transformers import SentenceTransformer
import config


embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)


connections.connect(host=config.MILVUS_HOST, port=config.MILVUS_PORT)
print("✅ Connected to Milvus")


id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000)
embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)

schema = CollectionSchema(
    fields=[id_field, text_field, embedding_field],
    description="Stores transcripts and documentation embeddings."
)


collection_name = "transcripts_and_docs"
collection = Collection(name=collection_name, schema=schema)


collection.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 100}})


def insert_text(text: str):
    """
    Inserts transcript or documentation text into Milvus after generating embeddings.
    """
    embedding = embedding_model.encode(text).tolist()
    collection.insert([[text], [embedding]])
    print("✅ Inserted Text into Milvus")

# ✅ Function to Retrieve Relevant Documents
def retrieve_docs(query: str, top_k=3):
    """
    Retrieves the most relevant documentation from Milvus using vector search.
    """
    query_embedding = embedding_model.encode(query).tolist()
    search_results = collection.search(query_embedding, "embedding", limit=top_k)
    return [res.entity for res in search_results]


    if __name__ == "__main__":
        #st the insert_text function
        insert_text("This is a test transcript.")                                                                           
        # Test the retrieve_docs function
        print(retrieve_docs("How to deploy a model in production?"))
