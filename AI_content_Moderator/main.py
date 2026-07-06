from typing import TypedDict, Annotated
from langchain_mistralai import ChatMistralAI
from langgraph.graph import StateGraph , START , END
from dotenv import load_dotenv
load_dotenv()

llm = ChatMistralAI(
    model="mistral-medium-3-5",
    temperature=0.1
)

def merge_score_dicts(existing:dict,newUpdate:dict)-> dict:
    if existing is None:
        return newUpdate
    return {**existing,**newUpdate}
    
class AnalyzerState(TypedDict):
    raw_text:str
    safety_score:Annotated[dict[str, int], merge_score_dicts]

#NODES
#TOXICITY NODE
def toxicity_checker_node(state: AnalyzerState) -> dict:
    """
    Evaluates the raw text for toxicity and assigns a severity score.

    This node acts as an expert content moderator. It analyzes the user's
    input text and determines a toxicity score ranging from 0 (completely safe)
    to 100 (extremely toxic/harmful). This score is merged into the global
    safety_score state dictionary.

    Args:
        state (AnalyzerState): The current state of the graph, containing the 'raw_text'.

    Returns:
        dict: A dictionary containing the updated 'safety_score' with the 'toxicity' key.
    """
    messages = [
        (
            "system",
            "You are an expert content moderation AI. Your task is to analyze the user's text for toxicity, "
            "hate speech, harassment, or harmful content. \n\n"
            "You must return ONLY a single integer between 0 and 100 representing the toxicity score:\n"
            "- 0: Completely safe, positive, or neutral.\n"
            "- 50: Borderline, slightly offensive or passive-aggressive.\n"
            "- 100: Extremely toxic, explicit hate speech, or severe harassment.\n\n"
            "Do not include any other text, explanations, or formatting. Just the integer."
        ),
        ("human", state["raw_text"])
    ]
    
    # Invoke the LLM with the structured messages
    res = llm.invoke(messages)
    
    # Parse the response and return it as an update to the safety_score dictionary
    try:
        score = int(res.content.strip())
        print(res.content.strip())
    except ValueError:
        # Fallback in case the LLM doesn't follow instructions perfectly
        score = 0
        
    return {"safety_score": {"toxicity": score}}
