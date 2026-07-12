from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mistralai import ChatMistralAI
from langchain.agents import create_agent
from dotenv import load_dotenv
import asyncio
load_dotenv()

async def main():
    client=MultiServerMCPClient(
        {
            "math":{
                "command":"python",
                "args":["mathserver.py"],
                "transport":"stdio"
            },
            "weather":{
                "url":"http://localhost:8000/mcp",
                "transport":"streamable_http"
            }
        }
    )

    import os
    os.environ["MISTRAL_API_KEY"]=os.getenv("MISTRAL_API_KEY")
    tools=await client.get_tools()
    model=ChatMistralAI(
        model="mistral-medium-3-5"
    )
    agent=create_agent(
        model,tools
    )
    math_res= await agent.ainvoke(
        {
            "messages":[{
                "role":"user",
                "content":"Whats the waether in islamabad?"
            }]
        }
    )

    print ("math result: ",math_res["messages"][-1].content)

asyncio.run(main())