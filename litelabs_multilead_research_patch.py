from pathlib import Path

exp_path = Path("/app/experimental_children_v1.py")
qa_path = Path("/app/qa_research.py")

exp = exp_path.read_text(encoding="utf-8")

anchor = "        # Mega53 is the routing brain:"
block = '''        # Experimental multi-singer branch. Experimental is currently
        # admin-only at the addon/usergroup layer, so run this path on every
        # Experimental extraction and keep it fail-open.
        try:
            from multilead_research import run_multilead_research
            multi_lead_report = run_multilead_research(
                stems["vocals"],
                experimental,
                track,
                progress=progress,
            )
            vocal_report["multi_lead"] = multi_lead_report
            timings["multi_lead_medleyvox"] = float(multi_lead_report.get("runtime_seconds") or 0.0)
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
            print(f"LiteLABS multi-lead separation skipped: {exc}", flush=True)

'''
if "multi_lead_medleyvox" not in exp:
    if anchor not in exp:
        raise RuntimeError("Could not locate Mega53 anchor for research multi-lead insertion")
    exp = exp.replace(anchor, block + anchor, 1)

# Teach the silent QA collector that A/B are two co-lead children rather than
# accidentally collapsing them into the generic lead_vocals label.
generic = '''            if "lead_vocals" in lower:
                label, model = "lead_vocals", "Becruily Karaoke (25% SW + 75% MelBand parent)"
'''
replacement = '''            if "lead_vocals_a" in lower:
                label, model = "lead_vocals_a", "MedleyVox vocals 238 (co-lead A)"
            elif "lead_vocals_b" in lower:
                label, model = "lead_vocals_b", "MedleyVox vocals 238 (co-lead B)"
            elif "lead_vocals" in lower:
                label, model = "lead_vocals", "Becruily Karaoke (25% SW + 75% MelBand parent)"
'''
if '"lead_vocals_a" in lower' not in exp:
    if generic not in exp:
        raise RuntimeError("Could not locate vocal QA mapping for multi-lead research")
    exp = exp.replace(generic, replacement, 1)


exp_path.write_text(exp, encoding="utf-8")

qa = qa_path.read_text(encoding="utf-8")
if '"multi_lead_children"' not in qa:
    child_anchor = '    "vocal_children": ("lead_vocals", "backing_vocals"),\n'
    if child_anchor not in qa:
        raise RuntimeError("Could not locate QA child groups")
    qa = qa.replace(
        child_anchor,
        child_anchor + '    "multi_lead_children": ("lead_vocals_a", "lead_vocals_b"),\n',
        1,
    )
    parent_anchor = '    "vocal_children": "vocals",\n'
    if parent_anchor not in qa:
        raise RuntimeError("Could not locate QA group parent map")
    qa = qa.replace(
        parent_anchor,
        parent_anchor + '    "multi_lead_children": "vocals",\n',
        1,
    )

qa_path.write_text(qa, encoding="utf-8")

check_exp = exp_path.read_text(encoding="utf-8")
check_qa = qa_path.read_text(encoding="utf-8")
assert "multi_lead_medleyvox" in check_exp
assert '"lead_vocals_a" in lower' in check_exp
assert '"multi_lead_children": ("lead_vocals_a", "lead_vocals_b")' in check_qa
print("LiteLABS Experimental MedleyVox multi-lead route applied")
