"""
This python script reads a PDF, extracts its text content, chunks into 
512 character pieces, and encodes each chunk into a vector using the
Qdrant vector database and SentenceTransformer 'all-MiniLM-L6-v2' model.
"""


from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer

import pymupdf

encoder = SentenceTransformer('all-MiniLM-L6-v2')

qdrant_client = QdrantClient(':memory:')

def read_pdf(file_path):
    """
    Reads a PDF file and extracts all text content.
    
    :param file_path: path to the PDF file
    :return: concatenated text from all pages
    """
    doc = pymupdf.open(file_path)

    full_text = ''
    for i, page in enumerate(doc): 
        text = page.get_text()
        full_text += text + '\n'

    return full_text

def chunk_text(text, chunk_size=512, overlap=50):
    """
    Chunks text into smaller piecces and returns a list of documents.

    :param text: full text to be chunked
    :param chunk_size: number of characters in each chunk
    :param overlap: number of overlapping characters between chunks

    Returns: a list of dictionary containing chunked text in the format
        documents = [
            {
                'text': 'chunked text here'
            },
            ...
        ]
    """
    documents = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        documents.append({'text': chunk})
        start += chunk_size - overlap

    return documents

def create_and_upload_in_mem_collection(
        collection_name='my_collection', documents=None
    ):
    """
    Creates an in-memory Qdrant collection and overwrites the previous
    one if it exists. The uploads the data to the in-memory collection.
    
    :param collection_name: name of the collection (optional)
    :param documents: list of documents to add to the collectoin. 
    """
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(),
            distance=models.Distance.COSINE,
        )
    )

    qdrant_client.upload_points(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=idx, vector=encoder.encode(doc['text']).tolist(), payload=doc
            )
            for idx, doc in enumerate(documents)
        ],
    )

def search_query(query, collection_name='my_collection', top_k=3):
    """
    Searches the in-memory Qdrant collection for the most similar documents
    to the query.

    :param query: search query string
    :param collection_name: name of the collection (optional)
    :param top_k: number of top similar documents to return (optional)

    Returns: tuple of (hits, search_results)
        hits: list of PointStruct objects with similarity scores
        search_results: list of text content from matched documents
    """
    hits = qdrant_client.query_points(
        collection_name=collection_name,
        query=encoder.encode(query).tolist(),
        limit=top_k,
    ).points
    
    search_results = [hit.payload['text'] for hit in hits]
    
    return hits, search_results

if __name__ == '__main__':
    full_text = read_pdf('input/yolov1/1506.02640v5.pdf')

    print('Reading and creating chunks...')
    documents = chunk_text(full_text, chunk_size=512, overlap=50)
    print(f"Total chunks created: {len(documents)}")
    
    print('Creating Qdrant collection...')

    create_and_upload_in_mem_collection(documents=documents)

    # Dummy search.
    hits, retrieved_list = search_query('YOLOv1 is a ')
    print('#' * 50)

    print('HITS:')
    print(hits)
    print('#' * 50)
    for hit in hits:
        print(hit.payload, 'score:', hit.score)
    print('#' * 50)
    print('RETRIEVED LIST:')
    print(retrieved_list)