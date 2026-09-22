from pathlib import Path

exp_path = Path('/app/experimental_children_v1.py')
text = exp_path.read_text(encoding='utf-8')

# This guard runs after LiteLABS-VX v2 has produced one lead/backing pair from
# the clean SW vocal parent. It only suppresses a backing export when that
# backing is demonstrably a gain-scaled copy of lead.
if 'vocal_duplicate_guard_v2' not in text:
    export_block = '''        lead_dest = experimental / f"{track}_lead_vocals.flac"
        backing_dest = experimental / f"{track}_backing_vocals.flac"
        _write_flac(lead_dest, best_lead, vocal_sr)
        _write_flac(backing_dest, best_backing, vocal_sr)

'''
    if export_block not in text:
        raise RuntimeError(
            'Could not locate LiteLABS-VX v2 vocal export block for duplicate guard'
        )

    guarded_export = '''        # vocal_duplicate_guard_v2
        def _vx_scaled_duplicate_analysis(lead_audio, backing_audio, sr):
            lead_arr = np.asarray(lead_audio, dtype=np.float32)
            backing_arr = np.asarray(backing_audio, dtype=np.float32)
            lead_mono = (
                np.mean(lead_arr, axis=1)
                if lead_arr.ndim > 1 else lead_arr
            )
            backing_mono = (
                np.mean(backing_arr, axis=1)
                if backing_arr.ndim > 1 else backing_arr
            )

            n = min(len(lead_mono), len(backing_mono))
            window = max(1, int(sr * 3.0))
            active_windows = 0
            duplicate_windows = 0
            duplicate_gain_db = []
            duplicate_corr = []
            duplicate_fit_db = []

            for start_idx in range(0, max(0, n - window + 1), window):
                lead_win = np.asarray(
                    lead_mono[start_idx:start_idx + window],
                    dtype=np.float64,
                )
                backing_win = np.asarray(
                    backing_mono[start_idx:start_idx + window],
                    dtype=np.float64,
                )
                lead_rms = float(
                    np.sqrt(np.mean(lead_win * lead_win) + 1e-12)
                )
                backing_rms = float(
                    np.sqrt(np.mean(backing_win * backing_win) + 1e-12)
                )
                if max(_db(lead_rms), _db(backing_rms)) < -55.0:
                    continue
                if min(_db(lead_rms), _db(backing_rms)) < -60.0:
                    continue

                active_windows += 1
                denom = float(
                    np.linalg.norm(lead_win) * np.linalg.norm(backing_win)
                )
                corr = (
                    abs(float(np.dot(lead_win, backing_win) / denom))
                    if denom > 1e-12 else 0.0
                )
                lead_energy = float(np.dot(lead_win, lead_win))
                scale = float(
                    np.dot(lead_win, backing_win)
                    / max(lead_energy, 1e-12)
                )
                fitted = backing_win - (scale * lead_win)
                fitted_rms = float(
                    np.sqrt(np.mean(fitted * fitted) + 1e-12)
                )
                fitted_relative_db = _db(
                    fitted_rms / max(backing_rms, 1e-12)
                )
                gain_db = _db(abs(scale))

                if corr >= 0.985 and fitted_relative_db <= -18.0:
                    duplicate_windows += 1
                    duplicate_gain_db.append(gain_db)
                    duplicate_corr.append(corr)
                    duplicate_fit_db.append(fitted_relative_db)

            duplicate_fraction = (
                float(duplicate_windows) / float(active_windows)
                if active_windows else 0.0
            )
            if duplicate_gain_db:
                p10, p90 = np.percentile(
                    np.asarray(duplicate_gain_db, dtype=np.float64),
                    [10, 90],
                )
                gain_spread_db = float(p90 - p10)
                median_gain_db = float(np.median(duplicate_gain_db))
                median_corr = float(np.median(duplicate_corr))
                median_fit_db = float(np.median(duplicate_fit_db))
            else:
                gain_spread_db = None
                median_gain_db = None
                median_corr = None
                median_fit_db = None

            suppress = bool(
                active_windows >= 6
                and duplicate_fraction >= 0.55
                and gain_spread_db is not None
                and gain_spread_db <= 3.0
            )
            return {
                "active_windows": int(active_windows),
                "duplicate_windows": int(duplicate_windows),
                "duplicate_fraction": round(duplicate_fraction, 6),
                "median_gain_db": (
                    round(median_gain_db, 3)
                    if median_gain_db is not None else None
                ),
                "gain_spread_db_p10_p90": (
                    round(gain_spread_db, 3)
                    if gain_spread_db is not None else None
                ),
                "median_correlation": (
                    round(median_corr, 6)
                    if median_corr is not None else None
                ),
                "median_scaled_fit_residual_db": (
                    round(median_fit_db, 3)
                    if median_fit_db is not None else None
                ),
                "backing_detected": not suppress,
                "suppressed_as_gain_scaled_duplicate": suppress,
            }

        duplicate_analysis = _vx_scaled_duplicate_analysis(
            best_lead,
            best_backing,
            vocal_sr,
        )

        lead_dest = experimental / f"{track}_lead_vocals.flac"
        backing_dest = experimental / f"{track}_backing_vocals.flac"
        _write_flac(lead_dest, best_lead, vocal_sr)
        vocal_files = [lead_dest.name]
        if duplicate_analysis["backing_detected"]:
            _write_flac(backing_dest, best_backing, vocal_sr)
            vocal_files.append(backing_dest.name)

'''
    text = text.replace(export_block, guarded_export, 1)

    # The v2 vocal patch initially creates both names. The duplicate guard now
    # owns that list, so remove the later unconditional assignment.
    unconditional_files = (
        '        vocal_files = [lead_dest.name, backing_dest.name]\n'
    )
    if unconditional_files in text:
        text = text.replace(unconditional_files, '', 1)

    report_anchor = '''            "backing_recipe": "Becruily Karaoke instrumental output from SW vocal parent",
            "best_stems_share_single_parent_pair": True,
'''
    if report_anchor not in text:
        raise RuntimeError(
            'Could not locate LiteLABS-VX v2 report block for duplicate guard'
        )
    report_update = '''            "backing_recipe": "Becruily Karaoke instrumental output from SW vocal parent",
            "backing_detected": bool(duplicate_analysis["backing_detected"]),
            "backing_suppressed_reason": (
                None if duplicate_analysis["backing_detected"]
                else "gain_scaled_duplicate_of_lead"
            ),
            "backing_duplicate_analysis": duplicate_analysis,
            "best_stems_share_single_parent_pair": True,
'''
    text = text.replace(report_anchor, report_update, 1)

# Keep QA model labels aligned with the new production route.
text = text.replace(
    'label, model = "lead_vocals", "Becruily Karaoke (25% SW + 75% MelBand parent)"',
    'label, model = "lead_vocals", "LiteLABS-VX v2 single-pass vocal parent route"',
)
text = text.replace(
    'label, model = "backing_vocals", "Becruily Karaoke (SW parent-minus-fast-lead)"',
    'label, model = "backing_vocals", "LiteLABS-VX v2 single-pass vocal parent route"',
)

exp_path.write_text(text, encoding='utf-8')

check = exp_path.read_text(encoding='utf-8')
compile(check, str(exp_path), 'exec')
assert 'vocal_duplicate_guard_v2' in check
assert 'suppressed_as_gain_scaled_duplicate' in check
assert '"backing_detected": bool(duplicate_analysis["backing_detected"])' in check
assert 'gain_scaled_duplicate_of_lead' in check
assert 'vocal_alt_parent' not in check
assert 'vocal_fast_karaoke' not in check
print('LiteLABS v2 duplicate backing-vocal guard applied')
