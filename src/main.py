from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_mistralai.chat_models import ChatMistralAI


if __name__ == '__main__':
# LLM
    run_local = True
    local_llm = local_llm = "gdisney/mistral-uncensored"
    if run_local == True:
        llm = ChatOllama(model=local_llm, format="json", temperature=0)
    else:
        llm = ChatMistralAI(
            model="mistral-medium", temperature=0, mistral_api_key=mistral_api_key
        )
print(llm.invoke("hello"))