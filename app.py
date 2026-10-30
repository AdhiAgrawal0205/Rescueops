"""Local tester for your agent.

  python app.py                  run all sample cases
  python app.py --case r2_p1     run one case
  python app.py --show           also print each scenario and your decision
  python app.py --test-api       check your Ollama setup with one small request
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WEATHER = {"clear": 1.0, "rain": 1.25, "storm": 1.5}
ACTIONS = ("dispatch", "wait", "reroute", "reject", "resupply")


def load_dotenv():
    """Tiny .env reader (local testing only). Never prints values."""
    path = os.path.join(HERE, ".env")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def stage_list(entry):
    return entry.get("stages", [entry])


def check(case, d):
    """Format + hard-safety self-check. This is NOT the competition score."""
    if not isinstance(d, dict):
        return ["output is not a dict (did choose_rescue return something?)"]
    errs = []
    if d.get("action") not in ACTIONS:
        errs.append("action must be one of " + ", ".join(ACTIONS))
    p = d.get("priority")
    if not isinstance(p, int) or isinstance(p, bool) or not 1 <= p <= 5:
        errs.append("priority must be an int from 1 to 5")
    if not isinstance(d.get("reason"), str) or not d["reason"].strip():
        errs.append("reason must be a non-empty string")
    if errs or d["action"] not in ("dispatch", "reroute"):
        return errs
    ms = {m["id"]: m for m in case["missions"]}
    mid = d.get("mission_id") or (case["missions"][0]["id"] if len(ms) == 1 else None)
    if mid not in ms:
        return ["mission_id missing or unknown"]
    m = ms[mid]
    alt = d["action"] == "reroute" or bool(d.get("uses_alternate_route"))
    dist = m["alternate_distance"] if alt else m["distance"]
    if alt and dist is None:
        return ["no alternate route exists for this mission"]
    if not alt and m["route_blocked"]:
        return ["BLOCKED ROUTE: you dispatched through a blocked road"]
    r = case["resources"]
    if math.ceil(dist / 10) > r["fuel"] or m["kits"] > r["kits"] or m["personnel"] > r["personnel"]:
        errs.append("RESOURCE VIOLATION: not enough fuel/kits/personnel")
    if math.ceil(dist * WEATHER.get(case["weather"], 1.0)) > m["deadline"]:
        errs.append("DEADLINE MISSED")
    return errs


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--case"); ap.add_argument("--show", action="store_true")
    ap.add_argument("--test-api", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, HERE)
    import agent
    with open(os.path.join(HERE, "examples", "sample_cases.json"), encoding="utf-8") as f:
        cases = json.load(f)
    if a.test_api:
        first = stage_list(cases[0])[0]
        out = agent.ask_ollama(first.get("agent_view", first))
        print("API OK, got decision:" if out else "No usable LLM decision (see messages above).", out or "")
        return
    bad = 0
    for e in cases:
        if a.case and e["case_id"] != a.case:
            continue
        for st in stage_list(e):
            try:
                d = agent.choose_rescue(json.loads(json.dumps(st.get("agent_view", st))))
            except Exception as ex:
                d, errs = None, [f"your agent crashed: {type(ex).__name__}: {ex}"]
            else:
                errs = check(st, d)
            tag = f"{e['case_id']} stage {st.get('stage', 0)}"
            print(("FAIL " if errs else "ok   ") + tag + (" -> " + "; ".join(errs) if errs else ""))
            bad += bool(errs)
            if a.show:
                print(json.dumps(st.get("agent_view", st))); print("  decision:", json.dumps(d))
    print(f"\n{bad} problem(s) found. (Passing here does not mean a high score.)")


if __name__ == "__main__":
    main()
