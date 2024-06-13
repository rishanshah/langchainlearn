from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import GPT4AllEmbeddings,HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings

def embed_document(url: str, run_local: bool):
    loader = WebBaseLoader(url)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=100
    )
    all_splits = text_splitter.split_documents(docs)

    if run_local == True:
        embedding = HuggingFaceBgeEmbeddings()
    else:
        embedding = MistralAIEmbeddings(mistral_api_key=mistral_api_key)

    vectorstore = Chroma.from_documents(
    documents=all_splits,
    collection_name="rag-chroma",
    embedding=embedding,
    )
    retriever = vectorstore.as_retriever()
    return retriever


# Load
# url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
# loader = WebBaseLoader(url)
# docs = loader.load()

# # Split
# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=500, chunk_overlap=100
# )
# all_splits = text_splitter.split_documents(docs)

# # Embed and index
# if run_local == "Yes":
#     embedding = GPT4AllEmbeddings()
# else:
#     embedding = MistralAIEmbeddings(mistral_api_key=mistral_api_key)

# # Index
# vectorstore = Chroma.from_documents(
#     documents=all_splits,
#     collection_name="rag-chroma",
#     embedding=embedding,
# )
# retriever = vectorstore.as_retriever()

if __name__ == '__main__':
    url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
    embed_document(url,run_local=True)
    print("Done")