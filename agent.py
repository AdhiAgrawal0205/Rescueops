"""RescueOps participant agent.

YOU build the agent with LangGraph + LangChain (+ Ollama). This file only gives you the function
definitions; write the bodies and the LangChain / LangGraph code yourself.

Scoring per case:  70% decision output  +  20% TOOL USAGE  +  10% speed.
  * Tool usage: the evaluator watches which LangChain tools your agent actually INVOKES.
    Make these four LangChain tools (decorate with `from langchain_core.tools import tool`)
    with EXACTLY these names, and call them with `.invoke({...})` while deciding:
        check_route, check_resources, check_deadline   (every case)
        parse_update_report                            (Round 5 cases that carry `update_report`)
  * Speed: Rounds 1-4 should answer in well under 0.3 s -> do NOT call the LLM there.
  * Rounds 1-4 run WITHOUT an API key. Your agent must work offline there.
  * Round 5 needs >= 1 successful Ollama call per case, made with the `requests` library
    (call_ollama below does that; use it inside your LangChain code, e.g. a RunnableLambda).
  * API key: NEVER write it in code. Set env vars OLLAMA_API_KEY / OLLAMA_MODEL locally; for Round 5
    you type your key + model in the organizer's submit form. Hard-coded keys get the file rejected.
"""
import json
import math
import os
import re

try:
    import requests
except ImportError:
    requests = None

# TODO: import what you need, e.g.
# from langchain_core.tools import tool
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnableLambda
# from langgraph.graph import StateGraph, END

OLLAMA_URL = "https://ollama.com/v1/chat/completions"
WEATHER = {"clear": 1.0, "rain": 1.25, "storm": 1.5}


# ---------------------------------------------------------------- provided helper (ready)
def call_ollama(prompt, timeout=20):
    """Send `prompt` (str) to your Ollama cloud model with `requests`. Returns the reply text, or "" if the
    key/model is missing or the call fails. Key + model come from env vars, never from code."""
    key, model = os.getenv("OLLAMA_API_KEY"), os.getenv("OLLAMA_MODEL")
    if not key or not model or requests is None:
        return ""
    try:
        r = requests.post(OLLAMA_URL, timeout=timeout,
                          headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                          json={"model": model, "temperature": 0,
                                "messages": [{"role": "user", "content": prompt}]})
        return r.json()["choices"][0]["message"]["content"] if r.status_code == 200 else ""
    except Exception:
        return ""


def ask_ollama(case, verbose=True):
    """Used by `python app.py --test-api` to check your setup. Returns the raw reply text or None."""
    text = call_ollama("Reply with the word OK and nothing else. Scenario: " + json.dumps(case)[:500])
    if not text and verbose:
        print("[ollama] no reply (check OLLAMA_API_KEY / OLLAMA_MODEL / model name / rate limit).")
    return text or None


# ---------------------------------------------------------------- 1) TOOLS: write these (make them LangChain @tool)
def check_route(mission, use_alternate=False):
    """Return {"ok": bool, "distance": int | None, "reason": str} for the direct (or alternate) route.
    A blocked direct route is not ok; alternate_distance may be None."""
    # TODO
    raise NotImplementedError


def check_resources(mission, distance, resources):
    """Return {"ok": bool, "fuel": int, "kits": int, "personnel": int} = what the trip NEEDS
    (fuel = ceil(km / 10)); ok only if `resources` covers all three."""
    # TODO
    raise NotImplementedError


def check_deadline(mission, distance, weather):
    """Return {"ok": bool, "minutes": int}; minutes = ceil(km * weather factor); ok if minutes <= deadline."""
    # TODO
    raise NotImplementedError


def parse_update_report(case, report, llm_text=""):
    """Round 5: apply the free-text update report (e.g. "Weather is now storm. Fuel is now 4.") to the stale
    case and return the UPDATED case. Verify any LLM hint yourself."""
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- 2) AGENT: build your LangGraph here
def build_graph():
    """TODO: build and return your compiled LangGraph (e.g. read -> analyze -> decide)."""
    raise NotImplementedError


def choose_rescue(case):
    """Called once per case state. Return a dict: action (dispatch|wait|reroute|reject|resupply),
    priority (int 1-5), reason (non-empty str); for dispatch/reroute also mission_id,
    uses_alternate_route, fuel_needed, kits_needed, personnel_needed. See README.md."""
    # TODO: run your graph and return its decision. The line below is only a placeholder.
    return {"action": "wait", "priority": 1, "reason": "placeholder: no strategy implemented yet"}
