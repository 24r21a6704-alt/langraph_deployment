import os
import sys
import io
import traceback
from typing import TypedDict, List, Optional

from fastapi import FastAPI
from langserve import add_routes
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
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

llm_flash = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=api_key,
)

llm = llm_flash

# ==========================================
# 2. STATE DEFINITION
# ==========================================

class CrewState(TypedDict):
    messages: List[BaseMessage]
    code: Optional[str]
    report: Optional[str]

# ==========================================
# 3. TOOLS
# ==========================================

@tool
def run_python_code(code: str) -> str:
    """Execute Python code and return stdout or traceback."""

    clean_code = (
        code.replace("```python", "")
        .replace("```", "")
        .strip()
    )

    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    try:
        local_scope = {}
        exec(clean_code, {}, local_scope)
        result = new_stdout.getvalue()
    except Exception:
        result = traceback.format_exc()
    finally:
        sys.stdout = old_stdout

    return result.strip() if result.strip() else "Success (no output)"


@tool
def generate_test_cases(task_description: str) -> str:
    """Generate QA test cases for a coding task."""

    prompt = (
        "You are a Senior QA Engineer. Generate 3 to 5 highly specific "
        f"test scenarios for the following coding task: {task_description}. "
        "Include edge cases and return them as a numbered list."
        "Do not include explanations, markdown, comments, or any extra text."
    )

    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)

# ==========================================
# 4. GRAPH NODES
# ==========================================

def real_time_developer(state: CrewState):
    task = state["messages"][-1].content

    prompt = (
        f"Write a clean Python script to solve this: {task}. "
        "Return ONLY the code with no explanation."
        "Do not include explanations, markdown, comments, or any extra text."
    )

    response = llm_flash.invoke(prompt)

    content = response.content
    if isinstance(content, list):
        code = (
            content[0].get("text", "")
            if isinstance(content[0], dict)
            else str(content[0])
        )
    else:
        code = str(content)

    return {"code": code}


def real_time_tester(state: CrewState):
    task = state["messages"][-1].content

    test_cases = generate_test_cases.invoke(task)
    execution = run_python_code.invoke({"code": state["code"]})

    report = (
        f"### EXECUTION OUTPUT\n{execution}\n\n"
        f"### TEST CASES\n{test_cases}"
    )

    return {"report": report}

# ==========================================
# 5. BUILD LANGGRAPH
# ==========================================

workflow = StateGraph(CrewState)

workflow.add_node("developer", real_time_developer)
workflow.add_node("tester", real_time_tester)

workflow.add_edge(START, "developer")
workflow.add_edge("developer", "tester")
workflow.add_edge("tester", END)

graph = workflow.compile()

# ==========================================
# 6. WRAP AS RUNNABLE
# ==========================================

def run_agent(task: str):
    initial_state = {
        "messages": [HumanMessage(content=task)]
    }

    result = graph.invoke(
        initial_state,
        config={"recursion_limit": 50},
    )

    return {
        "code": result.get("code", ""),
    }


agent = RunnableLambda(run_agent)

# ==========================================
# 7. FASTAPI + LANGSERVE
# ==========================================

app = FastAPI(
    title="LangGraph Dev-Test Crew",
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
# 8. RUN (LOCAL ONLY)
# ==========================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )
