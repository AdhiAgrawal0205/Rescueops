# Submission checklist
- [ ] `python app.py` runs with no crashes and no FAIL lines.
- [ ] `agent.py` defines `choose_rescue(case)` and returns a dict with `action`, `priority`, `reason`.
- [ ] Rounds 1-4 work with **no** API key and answer fast (< 2 s per case).
- [ ] Round 5: `python app.py --test-api` succeeds with your key/model, and your agent calls Ollama (via `requests`) on every Round 5 case, with a safe fallback.
- [ ] No API keys, `.env`, tokens or passwords anywhere in your files.
- [ ] Only `agent.py` (+ helper `.py`/`.json` files you import) in the ZIP. No `.venv`, no `__pycache__`.
- [ ] ZIP is under 1 MB and has `agent.py` at its top level (or in one folder).
- [ ] Team name is 2–40 letters/numbers/spaces/`-`/`_`.
