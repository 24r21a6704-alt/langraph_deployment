import os
import sys
import io
import traceback
from typing import TypedDict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

# ==========================================
# 0. APP SETUP
# ==========================================
app = FastAPI(title="LangGraph Dev-Test Crew API")

# ==========================================
# 1. LLM INITIALIZATION
# ==========================================
# On Render, set GEMINI_API_KEY as an Environment Variable
# (Dashboard -> your service -> Environment).
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    print("WARNING: GEMINI_API_KEY environment variable not set.")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # use a real, currently-available model name
    google_api_key=api_key,
)

# ==========================================
# 2. STATE DEFINITION
# ==========================================
class CrewState(TypedDict):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]


# ==========================================
# 3. TOOLS
# ==========================================
@tool
def run_python_code(code: str) -> str:
    """Execute python code and return the standard output or error trace."""
    if not isinstance(code, str):
        code = str(code)
    clean_code = code.replace("```python", "").replace("```", "").strip()

    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    try:
        local_scope = {}
        exec(clean_code, {}, local_scope)
        result = new_stdout.getvalue()
    except Exception:
        result = f"Execution Error:\n{traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout

    return result.strip() if result.strip() else "Success (no terminal output)"


@tool
def generate_test_cases(task_description: str) -> str:
    """Generate specific test scenarios for a given coding task."""
    prompt = (
        f"You are a Senior QA Engineer. Generate 3 to 5 highly specific test scenarios "
        f"for the following coding task: '{task_description}'.\n"
        f"Include standard cases and edge cases. Return them as a numbered list."
    )
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


def _extract_text(content):
    """Handle Gemini's content sometimes being a list of dicts."""
    if isinstance(content, list):
        if content and isinstance(content[0], dict):
            return content[0].get("text", "")
        return str(content[0]) if content else ""
    return str(content)


# ==========================================
# 4. GRAPH NODES (interactive input() removed)
# ==========================================
def real_time_developer(state: CrewState):
    task = state["messages"][-1].content
    dev_prompt = (
        f"Write a clean Python script to solve this: {task}. "
        f"Only return the code, no explanation or markdown formatting."
    )
    response = llm.invoke(dev_prompt)
    code_str = _extract_text(response.content)
    return {"code": code_str}


def real_time_tester(state: CrewState):
    task = state["messages"][-1].content

    test_cases = generate_test_cases.invoke(task)
    cases_str = _extract_text(test_cases)

    execution_result = run_python_code.invoke({"code": state["code"]})

    report = (
        f"### EXECUTION OUTPUT:\n{execution_result}\n\n"
        f"### TEST SCENARIOS EVALUATED:\n{cases_str}"
    )
    return {"report": report}


# ==========================================
# 5. GRAPH CONSTRUCTION (simplified: no interactive loop)
# ==========================================
rt_workflow = StateGraph(CrewState)
rt_workflow.add_node("developer", real_time_developer)
rt_workflow.add_node("tester", real_time_tester)

rt_workflow.add_edge(START, "developer")
rt_workflow.add_edge("developer", "tester")
rt_workflow.add_edge("tester", END)

rt_app = rt_workflow.compile()


# ==========================================
# 6. WEB ENDPOINTS
# ==========================================
class TaskRequest(BaseModel):
    task: str


@app.get("/")
def health():
    return {"status": "ok", "message": "LangGraph Dev-Test Crew API is running"}


@app.post("/run-task")
def run_task(request: TaskRequest):
    initial_state = {"messages": [HumanMessage(content=request.task)]}
    result = rt_app.invoke(initial_state, config={"recursion_limit": 50})
    return {
        "task": request.task,
        "code": result.get("code"),
        "report": result.get("report"),
    }
