from pathlib import Path

exp_path = Path("/app/experimental_children_v1.py")
qa_path = Path("/app/qa_research.py")

exp = exp_path.read_text(encoding="utf-8")

anchor = "        # Mega53 is the routing brain:"
block = '''        # MedleyVox is expensive and not yet reliable enough for every public
        # extraction. Keep it available for explicit research requests only.
        if bool(payload.get("enable_multi_lead")):
            try:
                from multilead_research import run_multilead_research
                multi_lead_report = run_multilead_research(
                    stems["vocals"],
                    experimental,
                    track,
                    progress=progress,
                )
                vocal_report["multi_lead"] = multi_lead_report
                timings["multi_lead_medleyvox"] = float(
                    multi_lead_report.get("runtime_seconds") or 0.0
                )
                vocal_files.extend([
                    name for name in multi_lead_report.get("files", [])
                    if name not in vocal_files
                ])
            except Exception as exc:
                vocal_report["multi_lead"] = {
                    "ok": False,
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                }
                print(
                    f"LiteLABS multi-lead separation skipped: {exc}",
                    flush=True,
                )
        else:
            vocal_report["multi_lead"] = {
                "ok": False,
                "ran": False,
                "reason": "disabled_by_default_fast_path",
            }
            timings["multi_lead_medleyvox"] = 0.0

'''
if "multi_lead_medleyvox" not in exp:
    if anchor not in exp:
        raise RuntimeError(
            "Could not locate routing anchor for research multi-lead insertion"
        )
    exp = exp.replace(anchor, block + anchor, 1)

# QA stem-name mapping is optional now because multi-lead is disabled on the
# production fast path. If a compatible mapping block exists, augment it; if
# not, do not make the whole production image depend on research-only labels.
if '"lead_vocals_a" in lower' not in exp:
    candidates = [
        '''            if "lead_vocals" in lower:
                label, model = "lead_vocals", "LiteLABS-VX v2 single-pass vocal parent route"
''',
        '''            if "lead_vocals" in lower:
                label, model = "lead_vocals", "Becruily Karaoke (25% SW + 75% MelBand parent)"
''',
        '''            if "lead_vocals" in lower:
                label, model = "lead_vocals", "BS-RoFormer Karaoke"
''',
    ]
    replacement = '''            if "lead_vocals_a" in lower:
                label, model = "lead_vocals_a", "MedleyVox vocals 238 (co-lead A)"
            elif "lead_vocals_b" in lower:
                label, model = "lead_vocals_b", "MedleyVox vocals 238 (co-lead B)"
            elif "lead_vocals" in lower:
                label, model = "lead_vocals", "LiteLABS-VX v2 single-pass vocal parent route"
'''
    for generic in candidates:
        if generic in exp:
            exp = exp.replace(generic, replacement, 1)
            break

exp_path.write_text(exp, encoding="utf-8")

qa = qa_path.read_text(encoding="utf-8")
if '"multi_lead_children"' not in qa:
    child_anchor = '    "vocal_children": ("lead_vocals", "backing_vocals"),\n'
    parent_anchor = '    "vocal_children": "vocals",\n'
    if child_anchor in qa:
        qa = qa.replace(
            child_anchor,
            child_anchor
            + '    "multi_lead_children": ("lead_vocals_a", "lead_vocals_b"),\n',
            1,
        )
    if parent_anchor in qa:
        qa = qa.replace(
            parent_anchor,
            parent_anchor + '    "multi_lead_children": "vocals",\n',
            1,
        )

qa_path.write_text(qa, encoding="utf-8")

check_exp = exp_path.read_text(encoding="utf-8")
check_qa = qa_path.read_text(encoding="utf-8")
compile(check_exp, str(exp_path), "exec")
compile(check_qa, str(qa_path), "exec")
assert "multi_lead_medleyvox" in check_exp
assert "disabled_by_default_fast_path" in check_exp
print("LiteLABS optional MedleyVox research route applied")
