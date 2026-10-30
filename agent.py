"""RescueOps participant agent  (LangGraph + LangChain tools + Ollama for Round 5).

STRUCTURE
---------

    solve_round1(case)   Basic Rescue          (direct route, 1 mission)
    solve_round2(case)   Blocked Routes        (blocked road + alternate route)
    solve_round3(case)   Limited Resources     (fuel / kits / personnel, resupply)
    solve_round4(case)   Critical Priorities   (many missions, best severity wins)
    solve_round5(case)   Adaptive RescueOps    (update_report + Ollama call)

"""
import json
import math
import os
import re
from typing import Any, Dict, List, TypedDict

try:
    import requests
except ImportError:  
    requests = None

from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph

OLLAMA_URL = "https://ollama.com/v1/chat/completions"


# =====================================================================================
# Provided helpers (Round 5 / --test-api use these; Rounds 1-4 never touch them)
# =====================================================================================
def call_ollama(prompt, timeout=20):
    """Send `prompt` to the Ollama cloud model with `requests`. Returns reply text or "" on any failure."""
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


# =====================================================================================
# ROUND 1 -- Basic Rescue
# One mission, clear weather, direct route only. Dispatch if feasible, otherwise reject.
# (Self-contained: own tools, own graph.)
# =====================================================================================
def _build_round1():
    W = {"clear": 1.0, "rain": 1.25, "storm": 1.5}

    @tool
    def check_route(mission: dict, use_alternate: bool = False) -> dict:
        """Return {'ok','distance','reason'} for the direct route of a mission (blocked => not ok)."""
        d = int(mission["distance"])
        if mission.get("route_blocked"):
            return {"ok": False, "distance": d, "reason": "direct route blocked"}
        return {"ok": True, "distance": d, "reason": "direct route open"}

    @tool
    def check_resources(mission: dict, distance: int, resources: dict) -> dict:
        """Return {'ok','fuel','kits','personnel'}: what the trip needs and whether we have it."""
        need = {"fuel": math.ceil(distance / 10), "kits": int(mission["kits"]),
                "personnel": int(mission["personnel"])}
        return {"ok": all(need[k] <= resources[k] for k in need), **need}

    @tool
    def check_deadline(mission: dict, distance: int, weather: str) -> dict:
        """Return {'ok','minutes'}: ceil(km * weather factor) compared with the deadline."""
        minutes = math.ceil(distance * W.get(weather, 1.0))
        return {"ok": minutes <= mission["deadline"], "minutes": minutes}

    class State(TypedDict, total=False):
        case: Dict[str, Any]
        options: List[Dict[str, Any]]
        decision: Dict[str, Any]

    def analyze(s):
        c, opts = s["case"], []
        for m in c["missions"]:
            rt = check_route.invoke({"mission": m, "use_alternate": False})
            rs = check_resources.invoke({"mission": m, "distance": rt["distance"], "resources": c["resources"]})
            dl = check_deadline.invoke({"mission": m, "distance": rt["distance"], "weather": c["weather"]})
            if rt["ok"] and rs["ok"] and dl["ok"]:
                opts.append({"m": m, "need": rs})
        return {"options": opts}

    def decide(s):
        opts = s["options"]
        if not opts:
            return {"decision": {"action": "reject", "priority": 1,
                                 "reason": "Round 1: mission not feasible (fuel or deadline), rejecting"}}
        o = max(opts, key=lambda x: x["m"]["severity"])
        n = o["need"]
        return {"decision": {"action": "dispatch", "priority": o["m"]["severity"],
                             "reason": "Round 1: direct route open, fuel and deadline OK",
                             "mission_id": o["m"]["id"], "uses_alternate_route": False,
                             "fuel_needed": n["fuel"], "kits_needed": n["kits"], "personnel_needed": n["personnel"]}}

    g = StateGraph(State)
    g.add_node("analyze", analyze)
    g.add_node("decide", decide)
    g.set_entry_point("analyze")
    g.add_edge("analyze", "decide")
    g.add_edge("decide", END)
    return g.compile()


_ROUND1_GRAPH = None


def solve_round1(case):
    global _ROUND1_GRAPH
    if _ROUND1_GRAPH is None:
        _ROUND1_GRAPH = _build_round1()
    return _ROUND1_GRAPH.invoke({"case": case})["decision"]


# =====================================================================================
# ROUND 2 -- Blocked Routes
# 1-2 missions. Never use a blocked road; try the alternate route (may not exist).
# (Self-contained: own tools, own graph.)
# =====================================================================================
def _build_round2():
    W = {"clear": 1.0, "rain": 1.25, "storm": 1.5}

    @tool
    def check_route(mission: dict, use_alternate: bool = False) -> dict:
        """Return {'ok','distance','reason'} for the direct or alternate route of a mission."""
        if use_alternate:
            d = mission.get("alternate_distance")
            if d is None:
                return {"ok": False, "distance": None, "reason": "no alternate route"}
            return {"ok": True, "distance": int(d), "reason": "alternate route open"}
        d = int(mission["distance"])
        if mission.get("route_blocked"):
            return {"ok": False, "distance": d, "reason": "direct route blocked"}
        return {"ok": True, "distance": d, "reason": "direct route open"}

    @tool
    def check_resources(mission: dict, distance: int, resources: dict) -> dict:
        """Return {'ok','fuel','kits','personnel'}: what the trip needs and whether we have it."""
        need = {"fuel": math.ceil(distance / 10), "kits": int(mission["kits"]),
                "personnel": int(mission["personnel"])}
        return {"ok": all(need[k] <= resources[k] for k in need), **need}

    @tool
    def check_deadline(mission: dict, distance: int, weather: str) -> dict:
        """Return {'ok','minutes'}: ceil(km * weather factor) compared with the deadline."""
        minutes = math.ceil(distance * W.get(weather, 1.0))
        return {"ok": minutes <= mission["deadline"], "minutes": minutes}

    class State(TypedDict, total=False):
        case: Dict[str, Any]
        options: List[Dict[str, Any]]
        decision: Dict[str, Any]

    def analyze(s):
        c, opts = s["case"], []
        for m in c["missions"]:
            for alt in (False, True):
                rt = check_route.invoke({"mission": m, "use_alternate": alt})
                if rt["distance"] is None:
                    continue
                rs = check_resources.invoke({"mission": m, "distance": rt["distance"], "resources": c["resources"]})
                dl = check_deadline.invoke({"mission": m, "distance": rt["distance"], "weather": c["weather"]})
                if rt["ok"] and rs["ok"] and dl["ok"]:
                    opts.append({"m": m, "alt": alt, "dist": rt["distance"], "need": rs})
        return {"options": opts}

    def decide(s):
        opts = s["options"]
        if not opts:
            return {"decision": {"action": "reject", "priority": 1,
                                 "reason": "Round 2: no safe route (blocked/no alternate/fuel/deadline), rejecting"}}
        o = max(opts, key=lambda x: (x["m"]["severity"], -x["dist"]))
        n = o["need"]
        return {"decision": {"action": "reroute" if o["alt"] else "dispatch", "priority": o["m"]["severity"],
                             "reason": "Round 2: best open route" + (" (alternate)" if o["alt"] else " (direct)"),
                             "mission_id": o["m"]["id"], "uses_alternate_route": o["alt"],
                             "fuel_needed": n["fuel"], "kits_needed": n["kits"], "personnel_needed": n["personnel"]}}

    g = StateGraph(State)
    g.add_node("analyze", analyze)
    g.add_node("decide", decide)
    g.set_entry_point("analyze")
    g.add_edge("analyze", "decide")
    g.add_edge("decide", END)
    return g.compile()


_ROUND2_GRAPH = None


def solve_round2(case):
    global _ROUND2_GRAPH
    if _ROUND2_GRAPH is None:
        _ROUND2_GRAPH = _build_round2()
    return _ROUND2_GRAPH.invoke({"case": case})["decision"]


# =====================================================================================
# ROUND 3 -- Limited Resources
# 2-3 missions, fuel / kits / personnel limited. If nothing fits: resupply (when possible) else reject.
# (Self-contained: own tools, own graph.)
# =====================================================================================
def _build_round3():
    W = {"clear": 1.0, "rain": 1.25, "storm": 1.5}

    @tool
    def check_route(mission: dict, use_alternate: bool = False) -> dict:
        """Return {'ok','distance','reason'} for the direct or alternate route of a mission."""
        if use_alternate:
            d = mission.get("alternate_distance")
            if d is None:
                return {"ok": False, "distance": None, "reason": "no alternate route"}
            return {"ok": True, "distance": int(d), "reason": "alternate route open"}
        d = int(mission["distance"])
        if mission.get("route_blocked"):
            return {"ok": False, "distance": d, "reason": "direct route blocked"}
        return {"ok": True, "distance": d, "reason": "direct route open"}

    @tool
    def check_resources(mission: dict, distance: int, resources: dict) -> dict:
        """Return {'ok','fuel','kits','personnel'}: what the trip needs and whether we have it."""
        need = {"fuel": math.ceil(distance / 10), "kits": int(mission["kits"]),
                "personnel": int(mission["personnel"])}
        return {"ok": all(need[k] <= resources[k] for k in need), **need}

    @tool
    def check_deadline(mission: dict, distance: int, weather: str) -> dict:
        """Return {'ok','minutes'}: ceil(km * weather factor) compared with the deadline."""
        minutes = math.ceil(distance * W.get(weather, 1.0))
        return {"ok": minutes <= mission["deadline"], "minutes": minutes}

    class State(TypedDict, total=False):
        case: Dict[str, Any]
        options: List[Dict[str, Any]]
        decision: Dict[str, Any]

    def analyze(s):
        c, opts = s["case"], []
        for m in c["missions"]:
            for alt in (False, True):
                rt = check_route.invoke({"mission": m, "use_alternate": alt})
                if rt["distance"] is None:
                    continue
                rs = check_resources.invoke({"mission": m, "distance": rt["distance"], "resources": c["resources"]})
                dl = check_deadline.invoke({"mission": m, "distance": rt["distance"], "weather": c["weather"]})
                if rt["ok"] and rs["ok"] and dl["ok"]:
                    opts.append({"m": m, "alt": alt, "dist": rt["distance"], "need": rs})
        return {"options": opts}

    def decide(s):
        c, opts = s["case"], s["options"]
        if not opts:
            if c.get("resupply_possible"):
                return {"decision": {"action": "resupply", "priority": 1,
                                     "reason": "Round 3: resources insufficient for every mission, resupply possible"}}
            return {"decision": {"action": "reject", "priority": 1,
                                 "reason": "Round 3: resources insufficient and no resupply, rejecting"}}
        # highest severity first; then the cheapest trip so scarce fuel/kits are not wasted
        o = max(opts, key=lambda x: (x["m"]["severity"], -x["need"]["fuel"], -x["dist"]))
        n = o["need"]
        return {"decision": {"action": "reroute" if o["alt"] else "dispatch", "priority": o["m"]["severity"],
                             "reason": "Round 3: most severe mission that our resources can cover",
                             "mission_id": o["m"]["id"], "uses_alternate_route": o["alt"],
                             "fuel_needed": n["fuel"], "kits_needed": n["kits"], "personnel_needed": n["personnel"]}}

    g = StateGraph(State)
    g.add_node("analyze", analyze)
    g.add_node("decide", decide)
    g.set_entry_point("analyze")
    g.add_edge("analyze", "decide")
    g.add_edge("decide", END)
    return g.compile()


_ROUND3_GRAPH = None


def solve_round3(case):
    global _ROUND3_GRAPH
    if _ROUND3_GRAPH is None:
        _ROUND3_GRAPH = _build_round3()
    return _ROUND3_GRAPH.invoke({"case": case})["decision"]


# =====================================================================================
# ROUND 4 -- Critical Priorities
# 3-5 missions, tight deadlines. Pick the best mission: severity first, then the safest/cheapest option.
# (Self-contained: own tools, own graph.)
# =====================================================================================
def _build_round4():
    W = {"clear": 1.0, "rain": 1.25, "storm": 1.5}

    @tool
    def check_route(mission: dict, use_alternate: bool = False) -> dict:
        """Return {'ok','distance','reason'} for the direct or alternate route of a mission."""
        if use_alternate:
            d = mission.get("alternate_distance")
            if d is None:
                return {"ok": False, "distance": None, "reason": "no alternate route"}
            return {"ok": True, "distance": int(d), "reason": "alternate route open"}
        d = int(mission["distance"])
        if mission.get("route_blocked"):
            return {"ok": False, "distance": d, "reason": "direct route blocked"}
        return {"ok": True, "distance": d, "reason": "direct route open"}

    @tool
    def check_resources(mission: dict, distance: int, resources: dict) -> dict:
        """Return {'ok','fuel','kits','personnel'}: what the trip needs and whether we have it."""
        need = {"fuel": math.ceil(distance / 10), "kits": int(mission["kits"]),
                "personnel": int(mission["personnel"])}
        return {"ok": all(need[k] <= resources[k] for k in need), **need}

    @tool
    def check_deadline(mission: dict, distance: int, weather: str) -> dict:
        """Return {'ok','minutes','slack'}: ceil(km * weather factor) compared with the deadline."""
        minutes = math.ceil(distance * W.get(weather, 1.0))
        return {"ok": minutes <= mission["deadline"], "minutes": minutes, "slack": mission["deadline"] - minutes}

    class State(TypedDict, total=False):
        case: Dict[str, Any]
        options: List[Dict[str, Any]]
        decision: Dict[str, Any]

    def analyze(s):
        c, opts = s["case"], []
        for m in c["missions"]:
            for alt in (False, True):
                rt = check_route.invoke({"mission": m, "use_alternate": alt})
                if rt["distance"] is None:
                    continue
                rs = check_resources.invoke({"mission": m, "distance": rt["distance"], "resources": c["resources"]})
                dl = check_deadline.invoke({"mission": m, "distance": rt["distance"], "weather": c["weather"]})
                if rt["ok"] and rs["ok"] and dl["ok"]:
                    opts.append({"m": m, "alt": alt, "dist": rt["distance"], "need": rs, "slack": dl["slack"]})
        return {"options": opts}

    def decide(s):
        c, opts = s["case"], s["options"]
        if not opts:
            if c.get("resupply_possible"):
                return {"decision": {"action": "resupply", "priority": 1,
                                     "reason": "Round 4: no mission is feasible now, resupply possible"}}
            return {"decision": {"action": "reject", "priority": 1,
                                 "reason": "Round 4: no mission is feasible, rejecting"}}
        # severity (score driver) > larger deadline slack (safer) > fewer resources used > shorter distance
        o = max(opts, key=lambda x: (x["m"]["severity"], x["slack"],
                                     -(x["need"]["fuel"] + x["need"]["kits"] + x["need"]["personnel"]), -x["dist"]))
        n = o["need"]
        return {"decision": {"action": "reroute" if o["alt"] else "dispatch", "priority": o["m"]["severity"],
                             "reason": "Round 4: highest-severity feasible mission among competing requests",
                             "mission_id": o["m"]["id"], "uses_alternate_route": o["alt"],
                             "fuel_needed": n["fuel"], "kits_needed": n["kits"], "personnel_needed": n["personnel"]}}

    g = StateGraph(State)
    g.add_node("analyze", analyze)
    g.add_node("decide", decide)
    g.set_entry_point("analyze")
    g.add_edge("analyze", "decide")
    g.add_edge("decide", END)
    return g.compile()


_ROUND4_GRAPH = None


def solve_round4(case):
    global _ROUND4_GRAPH
    if _ROUND4_GRAPH is None:
        _ROUND4_GRAPH = _build_round4()
    return _ROUND4_GRAPH.invoke({"case": case})["decision"]


# =====================================================================================
# ROUND 5 -- Adaptive RescueOps
# Scenario fields can be STALE; real changes arrive as free text in case["update_report"].
# Flow: read (Ollama call + parse_update_report) -> analyze -> decide.
# Needs >= 1 successful Ollama call (requests) per case.  (Self-contained: own tools, own graph.)
# =====================================================================================
def _build_round5():
    W = {"clear": 1.0, "rain": 1.25, "storm": 1.5}
    OLLAMA = "https://ollama.com/v1/chat/completions"

    @tool
    def check_route(mission: dict, use_alternate: bool = False) -> dict:
        """Return {'ok','distance','reason'} for the direct or alternate route of a mission."""
        if use_alternate:
            d = mission.get("alternate_distance")
            if d is None:
                return {"ok": False, "distance": None, "reason": "no alternate route"}
            return {"ok": True, "distance": int(d), "reason": "alternate route open"}
        d = int(mission["distance"])
        if mission.get("route_blocked"):
            return {"ok": False, "distance": d, "reason": "direct route blocked"}
        return {"ok": True, "distance": d, "reason": "direct route open"}

    @tool
    def check_resources(mission: dict, distance: int, resources: dict) -> dict:
        """Return {'ok','fuel','kits','personnel'}: what the trip needs and whether we have it."""
        need = {"fuel": math.ceil(distance / 10), "kits": int(mission["kits"]),
                "personnel": int(mission["personnel"])}
        return {"ok": all(need[k] <= resources[k] for k in need), **need}

    @tool
    def check_deadline(mission: dict, distance: int, weather: str) -> dict:
        """Return {'ok','minutes'}: ceil(km * weather factor) compared with the deadline."""
        minutes = math.ceil(distance * W.get(weather, 1.0))
        return {"ok": minutes <= mission["deadline"], "minutes": minutes}

    @tool
    def parse_update_report(case: dict, report: str, llm_text: str = "") -> dict:
        """Apply the free-text update report to the stale case and return the UPDATED case.
        The LLM hint (llm_text) is low-trust: it is only used when the report text mentions that field,
        and the deterministic regex rules below always override it."""
        c = json.loads(json.dumps(case))
        r = report or ""
        low = r.lower()
        try:
            mt = re.search(r"\{.*\}", llm_text or "", re.S)
            hint = json.loads(mt.group(0)) if mt else {}
            if hint.get("weather") in W and "weather" in low:
                c["weather"] = hint["weather"]
            f = hint.get("fuel")
            if isinstance(f, int) and not isinstance(f, bool) and f >= 0 and "fuel" in low:
                c["resources"]["fuel"] = f
        except (ValueError, TypeError, KeyError, AttributeError):
            pass
        # deterministic rules (reports are cumulative, so later sentences win)
        for x in re.finditer(r"weather is now (\w+)", r, re.I):
            if x.group(1).lower() in W:
                c["weather"] = x.group(1).lower()
        for x in re.finditer(r"fuel is now (\d+)", r, re.I):
            c["resources"]["fuel"] = int(x.group(1))
        for x in re.finditer(r"kits (?:are|is) now (\d+)", r, re.I):
            c["resources"]["kits"] = int(x.group(1))
        for x in re.finditer(r"personnel (?:are|is) now (\d+)", r, re.I):
            c["resources"]["personnel"] = int(x.group(1))
        for m in c["missions"]:
            mid = re.escape(m["id"])
            for x in re.finditer(r"route to %s is now (blocked|open)" % mid, r, re.I):
                m["route_blocked"] = x.group(1).lower() == "blocked"
            for x in re.finditer(r"%s severity is now (\d)" % mid, r, re.I):
                m["severity"] = int(x.group(1))
            for x in re.finditer(r"%s deadline is now (\d+)" % mid, r, re.I):
                m["deadline"] = int(x.group(1))
        return c

    def ollama_read(prompt_text):
        key, model = os.getenv("OLLAMA_API_KEY"), os.getenv("OLLAMA_MODEL")
        if not key or not model or requests is None:
            return ""
        try:
            resp = requests.post(OLLAMA, timeout=15,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                                 json={"model": model, "temperature": 0,
                                       "messages": [{"role": "user", "content": prompt_text}]})
            return resp.json()["choices"][0]["message"]["content"] if resp.status_code == 200 else ""
        except Exception:
            return ""

    llm_reader = RunnableLambda(ollama_read)

    class State(TypedDict, total=False):
        case: Dict[str, Any]
        live: Dict[str, Any]
        options: List[Dict[str, Any]]
        decision: Dict[str, Any]

    def read(s):
        case = s["case"]
        report = case.get("update_report") or ""
        # ALWAYS make the Ollama call (needed for the Round 5 score), even on stage 0 with no report.
        txt = llm_reader.invoke(
            "Read this rescue update report. Reply with ONLY a JSON object with optional keys "
            "weather (clear|rain|storm) and fuel (integer). Report: " + (report or "no report"))
        live = parse_update_report.invoke({"case": case, "report": report, "llm_text": txt}) if report else case
        return {"live": live}

    def analyze(s):
        c, opts = s["live"], []
        for m in c["missions"]:
            for alt in (False, True):
                rt = check_route.invoke({"mission": m, "use_alternate": alt})
                if rt["distance"] is None:
                    continue
                rs = check_resources.invoke({"mission": m, "distance": rt["distance"], "resources": c["resources"]})
                dl = check_deadline.invoke({"mission": m, "distance": rt["distance"], "weather": c["weather"]})
                if rt["ok"] and rs["ok"] and dl["ok"]:
                    opts.append({"m": m, "alt": alt, "dist": rt["distance"], "need": rs})
        return {"options": opts}

    def decide(s):
        c, opts = s["live"], s["options"]
        if not opts:
            if c.get("resupply_possible"):
                return {"decision": {"action": "resupply", "priority": 1,
                                     "reason": "Round 5: updated scenario has no feasible mission, resupply possible"}}
            return {"decision": {"action": "reject", "priority": 1,
                                 "reason": "Round 5: updated scenario has no feasible mission, rejecting"}}
        o = max(opts, key=lambda x: (x["m"]["severity"], -x["need"]["fuel"], -x["dist"]))
        n = o["need"]
        return {"decision": {"action": "reroute" if o["alt"] else "dispatch", "priority": o["m"]["severity"],
                             "reason": "Round 5: best feasible mission after applying the update report",
                             "mission_id": o["m"]["id"], "uses_alternate_route": o["alt"],
                             "fuel_needed": n["fuel"], "kits_needed": n["kits"], "personnel_needed": n["personnel"]}}

    g = StateGraph(State)
    g.add_node("read", read)
    g.add_node("analyze", analyze)
    g.add_node("decide", decide)
    g.set_entry_point("read")
    g.add_edge("read", "analyze")
    g.add_edge("analyze", "decide")
    g.add_edge("decide", END)
    return g.compile()


_ROUND5_GRAPH = None


def solve_round5(case):
    global _ROUND5_GRAPH
    if _ROUND5_GRAPH is None:
        _ROUND5_GRAPH = _build_round5()
    return _ROUND5_GRAPH.invoke({"case": case})["decision"]


# =====================================================================================
# ENTRY POINT -- only routes the case to the matching round solver. No shared logic.
# =====================================================================================
SOLVERS = {1: solve_round1, 2: solve_round2, 3: solve_round3, 4: solve_round4, 5: solve_round5}


def choose_rescue(case):
    """Called once per case state. Picks the solver by case["round"] and returns its decision dict."""
    rnd = case.get("round")
    if rnd not in SOLVERS:  # unknown/missing round: report present => 5, otherwise the most general offline solver
        rnd = 5 if case.get("update_report") else 4
    try:
        return SOLVERS[rnd](case)
    except Exception as e:  # never crash: a safe non-dispatch decision still scores
        return {"action": "reject", "priority": 1, "reason": f"safe fallback: round {rnd} solver failed ({type(e).__name__})"}
