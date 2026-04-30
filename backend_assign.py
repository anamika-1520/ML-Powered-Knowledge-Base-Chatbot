# backend/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated

app = FastAPI()

# 1. Setup LLM (Using gemma2-9b-it as requested) [cite: 16]
GROQ_API_KEY = "your_groq_api_token"
llm = ChatGroq(model_name="gemma2-9b-it", groq_api_key=GROQ_API_KEY)

# 2. Define LangGraph State [cite: 62]
class AgentState(TypedDict):
    input_text: str
    extracted_data: dict
    history: List[str]
    next_step: str

# 3. Define Tools (Minimum 5) [cite: 64]
def tool_log_interaction(state: AgentState):
    """Tool 1: Captures and saves interaction data [cite: 66]"""
    # Logic for entity extraction (HCP Name, Topics) goes here
    state['extracted_data']['status'] = "Logged"
    return state

def tool_edit_interaction(state: AgentState):
    """Tool 2: Allows modification of logged data [cite: 68]"""
    state['extracted_data']['status'] = "Updated"
    return state

def tool_analyze_sentiment(state: AgentState):
    """Tool 3: Infers HCP Sentiment (Positive/Neutral/Negative) [cite: 45]"""
    # Logic to call LLM for sentiment analysis
    return state

def tool_suggest_followups(state: AgentState):
    """Tool 4: Generates AI Suggested Follow-ups [cite: 51]"""
    return state

def tool_extract_entities(state: AgentState):
    """Tool 5: Identifies Materials and Samples mentioned [cite: 42]"""
    return state

# 4. Build the Graph [cite: 15]
workflow = StateGraph(AgentState)
workflow.add_node("process_input", tool_extract_entities)
workflow.add_node("log_data", tool_log_interaction)
workflow.set_entry_point("process_input")
workflow.add_edge("process_input", "log_data")
workflow.add_edge("log_data", END)
agent_app = workflow.compile()

# 5. API Endpoints
class InteractionRequest(BaseModel):
    message: str

@app.post("/chat-interaction")
async def handle_chat(request: InteractionRequest):
    initial_state = {"input_text": request.message, "extracted_data": {}, "history": []}
    result = agent_app.invoke(initial_state)
    return result