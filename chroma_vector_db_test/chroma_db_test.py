# # Ref: https://gist.github.com/jeffchuber/a9ebc0ad5c7b053b8d1c50449c07f893
# from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import Chroma
# from langchain.document_loaders import TextLoader

# # load the document and split it into chunks
# loader = TextLoader("./state_of_the_union.txt")
# documents = loader.load()

# # split it into chunks
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)

# # create the open-source embedding function
# embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# output_dir = "./db_metadata_v5"
# db = Chroma.from_documents(docs, embedding_function, persist_directory=output_dir)

# # query it
# query = "What did the president say about Ketanji Brown Jackson"
# docs = db.similarity_search(query)

# # print results
# print(docs[0].page_content)


# ============================================================================================
# Ref: https://github.com/chroma-core/chroma
#      https://docs.trychroma.com/production/administration/migration
#      https://blog.naver.com/pjt3591oo/223590941645?trackingCode=rss
import chromadb # setup Chroma in-memory, for easy prototyping. Can add persistence easily!

# client = chromadb.Client()
client = chromadb.PersistentClient(path="../perfume_recommendation_test/perfume_db")

collection = client.get_or_create_collection("new_collection")  # create_collection, get_collection, get_or_create_collection, delete_collection
# print(client.count_collections())
# print(collection.peek())

# Add docs to the collection. Can also update and delete. Row-based API coming soon!
collection.add(
    documents=["""크리드
    떼르 데르메스 오 드 뚜왈렛 (TERRE D'HERMES EDT)
    시트러스 / 우디 / 프레쉬 스파이시
    천진난만한 소년에서 어른이 되기까지의 과정을 연상시키는 향""", 
    """아쿠아 디 파르마
    콜로니아 에센자 오 드 코롱 (COLONIA ESSENZA EDC)
    시트러스 / 아로마틱
    웅축된 콜로니아에 우디함을 더해 시원하면서도 고급스럽지만 편안한 이미지 연출""", 
    """크리드
    어벤투스 오 드 퍼퓸 (AVENTUS EDP)
    프루티 / 스위트 / 레더
    용기와 힘, 비전, 그리고 성공을 기원하는 고급진 향"""], # Can handle tokenization, embedding, and indexing automatically. Can skip that and add your own embeddings as well
    metadatas=[{"brand": "creed"}, {"brand": "acqua di parma"}, {"brand": "creed"}], # Filter on these
    ids=["1","2","3"], # unique for each doc
)

# Query/search 1 most similar results. You can also .get by id
results = collection.query(
    query_texts=["고급진 향"],
    n_results=1,
    where={"brand": "creed"}, # optional filter
    # where_document={"$contains":"프루티"}  # optional filter
)

print(results)


# ============================================================================================
# import os
# import json
# from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain.schema import Document

# # Load perfume data
# perfume_data_path = "../perfume_recommendation_test/product.json"
# if not os.path.exists(perfume_data_path):
#     raise FileNotFoundError(f"Perfume data file not found: {perfume_data_path}")

# with open(perfume_data_path, "r", encoding="utf-8") as f:
#     perfume_data = json.load(f)

# if not perfume_data:
#     raise ValueError("Perfume data is empty.")

# # Prepare data for Chroma vector storage
# docs = []
# for perfume in perfume_data:
#     # Combine fields to form the input text
#     text = f"{perfume['brand']}\n{perfume['name_kr']} ({perfume['name_en']})\n{perfume['main_accord']}\n{perfume['content']}"
#     doc = Document(page_content=text)
#     docs.append(doc)

# # Create embeddings
# embedding_function = SentenceTransformerEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# # Build vector store
# output_dir = "./perfume_db"

# # Persist the vector DB with Chroma
# db = Chroma.from_documents(docs, embedding_function, persist_directory=output_dir)
# db.persist()

# print("Perfume data successfully stored in Chroma DB.")

# # Perform a query
# def query_perfumes(user_query):
#     """Search perfumes based on user query."""
#     results = db.similarity_search(user_query, k=5)
#     print("\nTop 5 Recommendations:")
#     for i, result in enumerate(results, 1):
#         print(f"{i}. {result.page_content}")

# if __name__ == "__main__":
#     # Example query
#     user_query = "어벤투스"
#     query_perfumes(user_query)