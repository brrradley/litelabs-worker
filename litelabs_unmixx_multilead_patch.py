from pathlib import Path

targets = [
    Path("/app/experimental_children_v1.py"),
    Path("/app/qa_research.py"),
]

replacements = {
    "multi_lead_medleyvox": "multi_lead_unmixx",
    "MedleyVox vocals 238 (co-lead A)": "UNMIXX chunked (co-lead A)",
    "MedleyVox vocals 238 (co-lead B)": "UNMIXX chunked (co-lead B)",
    "MedleyVox vocals 238 (Cyru5)": "UNMIXX chunked",
    "MedleyVox": "UNMIXX",
}

for path in targets:
    if not path.is_file():
        continue
    text = path.read_text(encoding="utf-8")
    for old, new in replacements.items():
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")
    compile(text, str(path), "exec")

exp = Path("/app/experimental_children_v1.py").read_text(encoding="utf-8")
assert "from multilead_research import run_multilead_research" in exp
assert "multi_lead_unmixx" in exp
assert "multi_lead_medleyvox" not in exp

print("LiteLABS UNMIXX multi-lead production labels applied")
