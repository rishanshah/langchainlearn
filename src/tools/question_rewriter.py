
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import hydra
from omegaconf import DictConfig
from pydantic import BaseModel
from langchain_community.chat_models import ChatOllama


class QuestionRewriter(BaseModel):
    chain: object = None

    def generate_chain(self,input_variables,prompt,llm):
        input_variables = list(input_variables)
        re_write_prompt = PromptTemplate(input_variables=input_variables,template=prompt)

        self.chain = re_write_prompt | llm | StrOutputParser()
        
    def invoke(self, question): 
     return self.chain.invoke({"question": question})
    
@hydra.main(version_base= None, config_path='../../config/llm',config_name='prompts')  
def main(cfg:DictConfig):
   qr = QuestionRewriter()
   local_llm = cfg.local_llm
   prompt = cfg.rewrite_prompt
   question = "agent memory"
   run_local = cfg.run_local
   input_variables = cfg.rewrite_prompt_input_variables
#    mistral_api_key = None
   if run_local == True:
        llm = ChatOllama(model=local_llm, format="json", temperature=0)
    # else:
    #     llm = ChatMistralAI(
    #         model="mistral-medium", temperature=0, mistral_api_key=mistral_api_key
    #     )

   qr.generate_chain(input_variables,prompt,llm)
   print(qr.invoke(question))
   
    
if __name__ == "__main__":
    main()