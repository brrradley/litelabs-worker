from pathlib import Path

exp_path = Path("/app/experimental_children_v1.py")
qa_path = Path("/app/qa_research.py")

exp = exp_path.read_text(encoding="utf-8")

anchor = "        # Mega53 is the routing brain:"
block = '''        # Research-only multi-singer branch. This never replaces the locked
        # production lead/backing outputs. It adds Lead Vocals A/B for A/B
        # listening when explicitly requested by the research payload.
        if bool(payload.get("research_multi_lead")):
            try:
                from multilead_research import run_multilead_research
                multi_lead_report = run_multilead_research(
                    stems["vocals"],
                    experimental,
                    track,
                    progress=progress,
                )
                vocal_report["multi_lead_research"] = multi_lead_report
                timings["multi_lead_medleyvox"] = float(multi_lead_report.get("runtime_seconds") or 0.0)
                vocal_files.extend([
                    name for name in multi_lead_report.get("files", [])
                    if name not in vocal_files
                ])
            except Exception as exc:
                vocal_report["multi_lead_research"] = {
                    "ok": False,
                    "research_only": True,
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                }
                print(f"LiteLABS research multi-lead separation skipped: {exc}", flush=True)

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


# Research pods are interactive rather than serverless, so allow a completed
# archive to be copied out of the TemporaryDirectory before it is destroyed.
if '"local_result_path": str(local_result_path) if local_result_path else None' not in exp:
    upload_anchor = '''        uploaded = False
        put_url = str(payload.get("result_put_url") or "").strip()
        if put_url:
'''
    upload_replacement = '''        uploaded = False
        local_result_path = None
        research_output_dir = str(payload.get("research_output_dir") or "").strip()
        if research_output_dir:
            import shutil
            output_root = Path(research_output_dir)
            output_root.mkdir(parents=True, exist_ok=True)
            local_result_path = output_root / archive.name
            shutil.copy2(archive, local_result_path)

        put_url = str(payload.get("result_put_url") or "").strip()
        if put_url:
'''
    if upload_anchor not in exp:
        raise RuntimeError("Could not locate research archive upload anchor")
    exp = exp.replace(upload_anchor, upload_replacement, 1)

    result_anchor = '''            "result_url": payload.get("result_public_url"),
            "root_parent_files":'''
    result_replacement = '''            "result_url": payload.get("result_public_url"),
            "local_result_path": str(local_result_path) if local_result_path else None,
            "root_parent_files":'''
    if result_anchor not in exp:
        raise RuntimeError("Could not locate research result return anchor")
    exp = exp.replace(result_anchor, result_replacement, 1)


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
assert "research_multi_lead" in check_exp
assert "multi_lead_medleyvox" in check_exp
assert '"lead_vocals_a" in lower' in check_exp
assert '"multi_lead_children": ("lead_vocals_a", "lead_vocals_b")' in check_qa
print("LiteLABS research MedleyVox multi-lead route applied")
