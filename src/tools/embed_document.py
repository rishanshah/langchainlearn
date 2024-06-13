from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from pydantic import BaseModel, Field

class DocumentEmbedder(BaseModel):
    vectorstore: Chroma = Field(default=None)
    mistral_api_key: str = None
    retriever: object = Field(default=None) 

    class Config:
        arbitrary_types_allowed = True

    def load_documents(self, url: str):
        loader = WebBaseLoader(url)
        return loader.load()

    def split_documents(self, docs):
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500, chunk_overlap=100
        )
        return text_splitter.split_documents(docs)

    def get_embeddings(self, run_local: bool):
        if run_local:
            return HuggingFaceBgeEmbeddings()
        else:
            return MistralAIEmbeddings(mistral_api_key=self.mistral_api_key)

    def create_vectorstore(self, all_splits, embedding):
        self.vectorstore = Chroma.from_documents(
            documents=all_splits,
            collection_name="rag-chroma",
            embedding=embedding,
        )

    def embed_document(self, url: str, run_local: bool):
        docs = self.load_documents(url)
        all_splits = self.split_documents(docs)
        embedding = self.get_embeddings(run_local)
        self.create_vectorstore(all_splits, embedding)
        self.retriever = self.vectorstore.as_retriever()

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
    embedder   = DocumentEmbedder()
    embedder.embed_document(url,run_local=True)
    print(embedder.retriever)