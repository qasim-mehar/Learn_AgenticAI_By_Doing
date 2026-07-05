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


def script_writter_node(state: pipleineState) -> dict:
    """
    LangGraph node to generate an engaging video/audio script based on the edited topic/outline.
    
    This node designs a structured script featuring a hook, main body sections, visual cues, 
    and a call to action, keeping the tone engaging and tailored to the topic.
    
    Args:
        state (pipleineState): The current state of the pipeline containing `edited_input`.
        
    Returns:
        dict: The updated state dictionary containing `script_text`.
    """
    messages = [
        (
            "system",
            "You are an expert, highly engaging scriptwriter for modern online video content (like YouTube or podcasts). "
            "Your task is to write a compelling script based on the provided topic/outline. "
            "Follow these structural and stylistic guidelines:\n\n"
            "- HOOK: Start with an attention-grabbing hook in the first 10 seconds. Raise a curiosity-inducing question or state a surprising fact.\n"
            "- BODY: Organize the content into clear, logical sections. Use smooth transitions between ideas.\n"
            "- CALL TO ACTION (CTA): Wrap up with a strong conclusion and a clear call to action (e.g., subscribing, leaving a comment, or checking out a link).\n"
            "- FORMATTING: Clearly separate [VISUAL DIRECTIONS] (in brackets) from the spoken NARRATION (in plain text) so it is easy to read.\n"
            "- TONE: Enthusiastic, conversational, clear, and direct. Use short, punchy sentences suited for natural vocal delivery."
        ),
        ("human", state["edited_input"]),
    ]
    res = llm.invoke(messages)
    return {"script_text": res.content.strip()}