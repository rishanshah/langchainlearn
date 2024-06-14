from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import hydra
from omegaconf import DictConfig
from embed_document import DocumentEmbedder
from langchain_community.chat_models import ChatOllama


from pydantic import BaseModel, Field

class RetrievalGrader(BaseModel):
    chain: object = None

    def create_grader_chain(self,
                            prompt:str,
                            llm,
                            input_variables: list[str]):
        input_variables = list(input_variables)
        prompt = PromptTemplate(input_variables=input_variables,template=prompt)
        self.chain = prompt | llm | JsonOutputParser()

    def retrieve_documents(self,
                           question,
                           retriever):
        docs = retriever.get_relevant_documents(question)
        doc_txt = docs[1].page_content
        return doc_txt
    
    def invoke(self,document,question):
        if self.chain == None:
            raise Exception
        else:
            return self.chain.invoke({"question": question, "document": document})

@hydra.main(version_base= None, config_path='../../config/llm',config_name='prompts')  
def main(cfg:DictConfig):
    run_local = cfg.run_local
    local_llm = cfg.local_llm
    prompt = cfg.prompt
    input_variables = cfg.input_variables
    grader = RetrievalGrader()
    url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
    question = "agent memory"
    # mistral_api_key = None

    if run_local == True:
        llm = ChatOllama(model=local_llm, format="json", temperature=0)
    # else:
    #     llm = ChatMistralAI(
    #         model="mistral-medium", temperature=0, mistral_api_key=mistral_api_key
    #     )

    embedder   = DocumentEmbedder()
    embedder.embed_document(url,run_local=True)
    print(embedder.retriever)

    grader.create_grader_chain(prompt=prompt,llm=llm,input_variables=input_variables)
    docs = grader.retrieve_documents(question,embedder.retriever)
    print(grader.invoke(docs,question))

if __name__ == '__main__':
    main()