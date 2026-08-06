import os
from typing import TypedDict, List, Optional

from fastapi import FastAPI
from langserve import add_routes
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

# ==========================================
# 1. LLM INITIALIZATION
# ==========================================

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError(
        "GOOGLE_API_KEY environment variable is not set. "
        "Add it in the Render dashboard under Environment."
    )

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=api_key,
)

# ==========================================
# 2. STATE DEFINITION
# ==========================================

class CrewState(TypedDict):
    messages: List[BaseMessage]
    code: Optional[str]

# ==========================================
# 3. DEVELOPER NODE
# ==========================================

def real_time_developer(state: CrewState):
    task = state["messages"][-1].content

    prompt = (
        f"Write a clean Python program for the following task:\n\n{task}\n\n"
        "Return ONLY executable Python code. "
        "Do not include explanations, markdown, comments, or any extra text."
    )

    response = llm.invoke(prompt)

    content = response.content
    if isinstance(content, list):
        code = (
            content[0].get("text", "")
            if isinstance(content[0], dict)
            else str(content[0])
        )
    else:
        code = str(content)

    return {"code": code.strip()}

# ==========================================
# 4. BUILD LANGGRAPH
# ==========================================

workflow = StateGraph(CrewState)

workflow.add_node("developer", real_time_developer)

workflow.add_edge(START, "developer")
workflow.add_edge("developer", END)

graph = workflow.compile()

# ==========================================
# 5. RUNNABLE FOR LANGSERVE
# ==========================================

def run_agent(task: str) -> str:
    initial_state = {
        "messages": [HumanMessage(content=task)]
    }

    result = graph.invoke(
        initial_state,
        config={"recursion_limit": 10},
    )

    # Return ONLY the code string
    return result.get("code", "")

agent = RunnableLambda(run_agent)

# ==========================================
# 6. FASTAPI + LANGSERVE
# ==========================================

app = FastAPI(
    title="LangGraph Code Generator",
    version="1.0",
)

@app.get("/")
def root():
    return {"status": "ok"}

add_routes(
    app,
    agent,
    path="/agent",
)

# ==========================================
# 7. RUN (LOCAL ONLY)
# ==========================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )
