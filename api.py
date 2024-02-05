import os,sys
import uvicorn
import time
import openai

from langchain import LLMChain, PromptTemplate
from langchain.llms import AzureOpenAI
from langchain.memory import ConversationBufferWindowMemory
import langchain
from langchain.callbacks import get_openai_callback

from fastapi import FastAPI,Request,APIRouter
from fastapi.params import Body
from pydantic import BaseModel

# API KEYs
os.environ["OPENAI_API_KEY"] = "API_KEY"
os.environ["OPENAI_API_TYPE"] = "Azure"
os.environ["OPENAI_API_VERSION"] = "2022-12-01"
os.environ["OPENAI_API_BASE"] = "https://{endpoint}.openai.azure.com"

router = APIRouter(tags=['PWC- Chat GPT Conversanation AI'])


class inputQuestion(BaseModel):
    askQuestion : str



template = """Assistant is a large language model trained by OpenAI.
Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.
Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
give response in html format only.

{history}

Human: {human_input}
Assistant:"""

prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)


chatgpt_chain = LLMChain(
    llm=AzureOpenAI( temperature=0.5,model ="gpt-3.5-turbo-16k",deployment_name="NAME"),
    prompt=prompt,
    verbose=True,
    memory=ConversationBufferWindowMemory(k=2,memory_key="history",return_messages=True),

)

@router.post("/conversationAI")
async def askQuery(request:Request,inputQ:inputQuestion):
    try:
        query = inputQ.askQuestion
        with get_openai_callback() as cb:
            start=time.time()
            output=chatgpt_chain.predict(human_input= query)
            end=time.time()
            time_taken=f"time={end-start:.2f}seconds"
            return output,cb,time_taken
         
        
    except Exception as ex:
        print(f"Error identified -- {ex}")
        return "Error",ex
