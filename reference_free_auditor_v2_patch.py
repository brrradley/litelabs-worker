from pathlib import Path

path = Path('/app/reference_free_stem_auditor.py')
text = path.read_text()

old = '''        retry_queue.sort(key=lambda item: (-item["priority"], item["stem"]))
        missing = [role for role in PRIMARY_STEMS if role not in loaded]
        overall_status = "pass"
        if any(item["status"] == "retry" for item in stems) or reconstruction_error_db > -15:
            overall_status = "retry_recommended"
        elif retry_queue or missing or reconstruction_error_db > -25:
            overall_status = "review_recommended"
'''

new = '''        missing = [role for role in PRIMARY_STEMS if role not in loaded]
        absorption_suspected = bool(missing and reconstruction_error_db <= -25 and reconstruction_corr >= 0.98)
        for role in missing:
            reasons = [f"{role} output is missing from the delivered pack"]
            if absorption_suspected:
                reasons.append("the remaining stems still reconstruct the source closely, so this material was probably absorbed into another stem")
            retry_queue.append({
                "stem": role,
                "priority": 3,
                "suggested_route": "recover_raw_model_output_then_compare_specialist_candidate",
                "reasons": reasons,
                "failure_type": "missing_primary_output",
            })

        if absorption_suspected:
            for item in stems:
                if item["stem"] in {"guitar", "drums", "vocals", "bass"}:
                    item["absorption_risk"] = "possible"
                    item["reasons"].append("missing stem material may have been assigned here because the reduced stem set still reconstructs the source")
                    if item["status"] == "pass":
                        item["status"] = "review"
                        item["suggested_route"] = "compare_against_full_six_stem_raw_output"

        retry_queue.sort(key=lambda item: (-item["priority"], item["stem"]))
        overall_status = "pass"
        if missing or any(item["status"] == "retry" for item in stems) or reconstruction_error_db > -15:
            overall_status = "retry_recommended"
        elif retry_queue or reconstruction_error_db > -25:
            overall_status = "review_recommended"
'''

if old not in text:
    raise SystemExit('target block not found')
text = text.replace(old, new)
text = text.replace('"schema_version": 1,', '"schema_version": 2,')
text = text.replace('            "missing_primary_stems": missing,', '            "missing_primary_stems": missing,\n            "absorption_suspected": absorption_suspected,')
path.write_text(text)
