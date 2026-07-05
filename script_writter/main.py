import os
import sys
from dotenv import load_dotenv
from typing import TypedDict
from langchain_mistralai import ChatMistralAI 
from rich import print

sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

class pipleineState(TypedDict):
    raw_input:str
    edited_input:str
    script_text:str
    roman_urdu:str


llm=ChatMistralAI(
    model="mistral-medium-3-5",
)

def text_editor_node(state: pipleineState) -> dict:
    """
    LangGraph node to clean up and edit the raw input text.
    
    This node corrects grammatical errors, improves sentence structure, and
    polishes the vocabulary of the raw input while preserving its original meaning.
    
    Args:
        state (pipleineState): The current state of the pipeline containing `raw_input`.
        
    Returns:
        dict: The updated state dictionary containing `edited_input`.
    """
    messages = [
        (
            "system",
            "You are a professional copyeditor. Your job is to clean up, refine, and polish the user's input text. "
            "Correct all spelling mistakes, grammatical errors, and punctuation issues. "
            "Improve sentence flow and phrasing, ensuring it is clear and professional while preserving the original message, tone, and meaning. "
            "Do not add any introductory or concluding remarks, explanations, or formatting. "
            "Return only the edited text."
        ),
        ("human", state["raw_input"]),
    ]
    res = llm.invoke(messages)
    return {"edited_input": res.content.strip()}