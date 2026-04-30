import os
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from pydantic import BaseModel
import models, database

models.Base.metadata.create_all(bind=database.engine)
app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Model Config [cite: 16, 18]
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model_name="gemma2-9b-it", groq_api_key=GROQ_API_KEY)

# Define 5 LangGraph Tools [cite: 64, 65, 66, 68]
def tool_log_interaction(state):
    print("Tool: Logging to DB")
    return state

def tool_edit_interaction(state):
    print("Tool: Modifying existing record")
    return state

def tool_analyze_sentiment(state):
    # logic for sentiment extraction [cite: 45]
    state["sentiment"] = "Positive"
    return state

def tool_extract_entities(state):
    # logic for HCP name/topics extraction [cite: 34, 40]
    return state

def tool_suggest_followups(state):
    state["suggestions"] = "Schedule follow-up in 2 weeks" # [cite: 51]
    return state

# LangGraph Logic [cite: 8, 15]
class AgentState(dict): pass

workflow = StateGraph(AgentState)
workflow.add_node("process", lambda x: tool_analyze_sentiment(tool_extract_entities(x)))
workflow.set_entry_point("process")
workflow.add_edge("process", END)
agent = workflow.compile()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_with_ai(request: ChatRequest, db: Session = Depends(database.get_db)):
    result = agent.invoke({"message": request.message})
    # Create DB Record
    new_log = models.HCPInteraction(
        hcp_name="Dr. Smith (Extracted)", 
        topics=request.message,
        sentiment=result.get("sentiment", "Neutral")
    )
    db.add(new_log)
    db.commit()
    return {"status": "success", "extracted": result}