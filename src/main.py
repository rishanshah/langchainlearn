from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_mistralai.chat_models import ChatMistralAI
import hydra
from omegaconf import DictConfig, OmegaConf

mistral_api_key = None

@hydra.main(version_base= None, config_path='../config/llm',config_name='config')
def main(cfg:DictConfig):
    # LLM
    run_local = cfg.run_local
    local_llm = cfg.local_llm
    if run_local == True:
        llm = ChatOllama(model=local_llm, format="json", temperature=0)
    else:
        llm = ChatMistralAI(
            model="mistral-medium", temperature=0, mistral_api_key=mistral_api_key
        )
    print(llm.invoke("hello"))

if __name__ == '__main__':
    main()