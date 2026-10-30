"""RescueOps local tester with separate JSON files for each round.

Usage:
  python app.py                         # test all rounds
  python app.py --round 3               # test only Round 3
  python app.py --case r2_p1 --show     # test one case
  python app.py --round 5 --show        # show all Round 5 stages
  python app.py --test-api              # test participant Ollama function
"""
import argparse, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, "example")
WEATHER = {"clear": 1.0, "rain": 1.25, "storm": 1.5}
ACTIONS = ("dispatch", "wait", "reroute", "reject", "resupply")
ROUND_FILES = {
    1: "round_1_basic_rescue.json",
    2: "round_2_blocked_routes.json",
    3: "round_3_limited_resources.json",
    4: "round_4_critical_priorities.json",
    5: "round_5_adaptive_rescueops.json",
}

def load_cases(round_filter=None):
    cases = []
    rounds = [round_filter] if round_filter else range(1, 6)
    for r in rounds:
        path = os.path.join(EXAMPLES, ROUND_FILES[r])
        with open(path, encoding="utf-8") as f:
            cases.extend(json.load(f))
    return cases

def stage_list(entry):
    return entry.get("stages", [entry])

def check(case, decision):
    if not isinstance(decision, dict):
        return ["output is not a JSON object/dict"]
    errs = []
    action = decision.get("action")
    if action not in ACTIONS:
        errs.append("action must be one of " + ", ".join(ACTIONS))
    priority = decision.get("priority")
    if not isinstance(priority, int) or isinstance(priority, bool) or not 1 <= priority <= 5:
        errs.append("priority must be an integer from 1 to 5")
    if not isinstance(decision.get("reason"), str) or not decision["reason"].strip():
        errs.append("reason must be non-empty text")
    if errs or action not in ("dispatch", "reroute"):
        return errs

    missions = {m["id"]: m for m in case["missions"]}
    mid = decision.get("mission_id") or (case["missions"][0]["id"] if len(missions) == 1 else None)
    if mid not in missions:
        return ["mission_id is missing or unknown"]
    m = missions[mid]
    use_alt = action == "reroute" or bool(decision.get("uses_alternate_route"))
    distance = m["alternate_distance"] if use_alt else m["distance"]
    if use_alt and distance is None:
        return ["no alternate route exists"]
    if not use_alt and m["route_blocked"]:
        return ["unsafe: direct route is blocked"]
    resources = case["resources"]
    fuel_needed = math.ceil(distance / 10)
    if (fuel_needed > resources["fuel"] or m["kits"] > resources["kits"]
            or m["personnel"] > resources["personnel"]):
        errs.append("resource limit exceeded")
    travel_time = math.ceil(distance * WEATHER.get(case.get("weather", "clear"), 1.0))
    if travel_time > m["deadline"]:
        errs.append("deadline cannot be met")
    return errs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, choices=range(1, 6))
    parser.add_argument("--case")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--test-api", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, HERE)
    import agent

    if args.test_api:
        fn = getattr(agent, "call_ollama", None)
        if not callable(fn):
            print("Implement call_ollama(prompt) in agent.py first.")
            return
        try:
            result = fn('Return the word READY only.')
            print("Ollama response received." if result else "No response received.")
        except Exception as exc:
            print("Ollama test failed:", type(exc).__name__, str(exc))
        return

    cases = load_cases(args.round)
    selected = 0
    failures = 0
    for entry in cases:
        if args.case and entry["case_id"] != args.case:
            continue
        selected += 1
        for stage_case in stage_list(entry):
            view = stage_case.get("agent_view", stage_case)
            try:
                decision = agent.choose_rescue(json.loads(json.dumps(view)))
                errors = check(stage_case, decision)
            except Exception as exc:
                decision, errors = None, [f"agent crashed: {type(exc).__name__}: {exc}"]
            tag = f'{entry["case_id"]} / stage {stage_case.get("stage", 0)}'
            print(("FAIL " if errors else "PASS ") + tag)
            for error in errors:
                print("  - " + error)
            if args.show:
                print("  INPUT:", json.dumps(view, ensure_ascii=False))
                print("  OUTPUT:", json.dumps(decision, ensure_ascii=False))
            failures += bool(errors)
    if selected == 0:
        print("No matching case found. Check --round / --case.")
        return
    print(f"\nSummary: {selected} case file(s), {failures} failed stage(s).")
    print("Note: this is a local format/safety checker, not the official competition score.")

if __name__ == "__main__":
    main()
