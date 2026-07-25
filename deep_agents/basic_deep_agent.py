from typing import Literal 
from langchain_mistralai.chat_models import ChatMistralAI
from deepagents import create_deep_agent
from tavily import TavilyClient
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["TAVILY_API_KEY"]=os.getenv("TAVILY_API_KEY")
os.environ["MISTRAL_API_KEY"]=os.getenv("MISTRAL_API_KEY")

tavily_client =TavilyClient(api_key=os.getenv("TAVILY_API_KEY") )

def web_search(query:str,max_results:int=5, topic:Literal["sports","finance","general"]="general",include_raw_content:bool=False):
    "Run a web search"
    return tavily_client.search(query=query, max_results=max_results,include_raw_content=include_raw_content,topic=topic)


model=ChatMistralAI(
    model="mistral-medium-3-5"

)

deepagent=create_deep_agent(
    model=model,
    tools=[web_search]
)

result=deepagent.invoke(
    {"messages":[{"role":"user","content":"What is a deepagent and how it is different from a tranditional one"}]}
)

print(result)