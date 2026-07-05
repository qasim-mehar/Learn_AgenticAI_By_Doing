from rich import print
import os
import sys
from dotenv import load_dotenv
from typing import TypedDict
from langchain_mistralai import ChatMistralAI 
from langgraph.graph import StateGraph, START, END

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

def roman_urdu_node(state: pipleineState) -> dict:
    """
    LangGraph node to translate the English script into conversational Roman Urdu.
    
    This node translates the spoken narration of the English script into natural,
    conversational Roman Urdu (Urdu written in the Latin alphabet) while keeping
    visual directions in English and maintaining the original structure.
    
    Args:
        state (pipleineState): The current state of the pipeline containing `script_text`.
        
    Returns:
        dict: The updated state dictionary containing `roman_urdu`.
    """
    messages = [
        (
            "system",
            "You are an expert translator specializing in translating English video scripts into conversational Roman Urdu "
            "(Urdu written in the Latin/English alphabet).\n"
            "Your task is to translate the provided English script. Follow these rules:\n\n"
            "- TRANSLATION: Translate all spoken NARRATION into natural, fluid Roman Urdu. Use words and phrasing that sound "
            "natural when spoken aloud (everyday spoken Urdu, not overly formal or literal translations).\n"
            "- ENGLISH LOAN WORDS: Keep common modern English words and technical terms in English (e.g., 'AI', 'computer', "
            "'subscribe', 'algorithm', 'link', 'video') written in standard English.\n"
            "- VISUAL DIRECTIONS: Keep all [VISUAL DIRECTIONS] inside the brackets in English as they are for visual reference. "
            "Do not translate text inside brackets.\n"
            "- STRUCTURE: Maintain the exact structure, formatting, and line breaks of the original script. Return only the "
            "translated script."
        ),
        ("human", state["script_text"]),
    ]
    res = llm.invoke(messages)
    return {"roman_urdu": res.content.strip()}


#adding nodes in graph
graph=StateGraph(pipleineState)

graph.add_node("TextEditor",text_editor_node)
graph.add_node("ScriptWritter",script_writter_node)
graph.add_node("Translator",roman_urdu_node)

#connecting nodes using edges

graph.add_edge(START, "TextEditor")
graph.add_edge("TextEditor", "ScriptWritter")
graph.add_edge("ScriptWritter","Translator")
graph.add_edge("Translator", END)

app=graph.compile()
final_script=app.invoke({
    "raw_input": "how to focus in world of distraction... smartphone are ruining focus. we scroll instagram, tiktok all day. dopamine detox is needed. 1st step: turn off notifications. 2nd step: keep phone in other room while working. 3rd step: replace scrolling with reading or walking. it helps reclaim brain power. lets do a 24 hour challenge."
})
print(final_script["roman_urdu"])