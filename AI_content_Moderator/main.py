from typing import TypedDict, Annotated
from langchain_mistralai import ChatMistralAI
from langgraph.graph import StateGraph , START , END
from dotenv import load_dotenv
from rich import print
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

#CULTURAL NODE
def cultural_sensitivity_node(state: AnalyzerState) -> dict:
    """
    Evaluates the raw text for cultural insensitivity, bias, and stereotypes.

    This node analyzes the input text for potentially offensive language regarding
    race, religion, gender, or cultural heritage. It returns a score from 
    0 (highly sensitive/inclusive) to 100 (highly insensitive/biased).

    Args:
        state (AnalyzerState): The current state of the graph, containing the 'raw_text'.

    Returns:
        dict: A dictionary containing the updated 'safety_score' with the 'cultural_insensitivity' key.
    """
    messages = [
        (
            "system",
            "You are an expert AI in diversity, equity, and inclusion (DEI). Your task is to analyze the user's text for "
            "cultural insensitivity, bias, stereotyping, or exclusionary language. \n\n"
            "You must return ONLY a single integer between 0 and 100 representing the insensitivity score:\n"
            "- 0: Completely inclusive, respectful, and free of bias.\n"
            "- 50: Contains mild stereotypes or borderline insensitive phrasing.\n"
            "- 100: Highly offensive, blatant prejudice, or severe cultural insensitivity.\n\n"
            "Do not include any other text, explanations, or formatting. Just the integer."
        ),
        ("human", state["raw_text"])
    ]
    
    res = llm.invoke(messages)
    
    try:
        score = int(res.content.strip())
        
    except ValueError:
        score = 0
        
    return {"safety_score": {"cultural_insensitivity": score}}

#COPYRIGHT NODE
def copyright_checker_node(state: AnalyzerState) -> dict:
    """
    Evaluates the raw text for potential copyright infringement or IP violations.

    This node checks if the text heavily borrows from protected works, quotes 
    excessive amounts of lyrics/scripts without attribution, or attempts to 
    plagiarize known content. It returns a risk score from 0 to 100.

    Args:
        state (AnalyzerState): The current state of the graph, containing the 'raw_text'.

    Returns:
        dict: A dictionary containing the updated 'safety_score' with the 'copyright_risk' key.
    """
    messages = [
        (
            "system",
            "You are an expert intellectual property (IP) and copyright analyst AI. Your task is to analyze the user's text "
            "for potential copyright infringement, plagiarism, or unauthorized distribution of protected works. \n\n"
            "You must return ONLY a single integer between 0 and 100 representing the copyright risk score:\n"
            "- 0: Completely original text or clearly fair use.\n"
            "- 50: Contains significant recognizable quotes or borderline derivative work.\n"
            "- 100: Blatant plagiarism, sharing full protected lyrics/scripts, or severe IP violation.\n\n"
            "Do not include any other text, explanations, or formatting. Just the integer."
        ),
        ("human", state["raw_text"])
    ]
    
    res = llm.invoke(messages)
    
    try:
        score = int(res.content.strip())
        print(f"Copyright Score: {score}")
    except ValueError:
        score = 0
        
    return {"safety_score": {"copyright_risk": score}}

#Building graph
graph_builder=StateGraph(AnalyzerState)

#CRAETING NODES IN GRAPH
graph_builder.add_node("toxicity_node", toxicity_checker_node)
graph_builder.add_node("cultural_node", cultural_sensitivity_node)
graph_builder.add_node("copyright_node", copyright_checker_node)

#CONNECTING NODES USING EDGES

graph_builder.add_edge(START, "toxicity_node")
graph_builder.add_edge(START, "cultural_node")
graph_builder.add_edge(START, "copyright_node")

graph_builder.add_edge("toxicity_node", END)
graph_builder.add_edge("cultural_node", END)
graph_builder.add_edge("copyright_node" ,END)

graph=graph_builder.compile()

if __name__ == "__main__":
  
    test_text = (
        "You are absolutely pathetic and I hate your guts. "
        "Also, as Mickey Mouse always says in Disney's famous movie: 'I am going to sue you!' "
        "And another thing, all people from Mars are lazy and smell bad."
    )
    
    print("--- Starting Content Analysis ---")
    print(f"Text: {test_text}\n")
    
    # Run the LangGraph
    # We only need to provide the raw_text. 
    # The safety_score dictionary will be built dynamically by our merge_score_dicts function.
    result = graph.invoke({"raw_text": test_text})
    
    print("\n--- Final Analysis Results ---")
    
    # We use json.dumps just to print the dictionary beautifully
    import json
    print(json.dumps(result.get("safety_score", {}), indent=4))