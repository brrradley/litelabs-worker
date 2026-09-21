from pathlib import Path

exp_path = Path('/app/experimental_children_v1.py')
text = exp_path.read_text(encoding='utf-8')

old = '''        lead_n = min(len(quality_lead), blend_n)
        best_lead = np.asarray(quality_lead[:lead_n], dtype=np.float32)
        lead_dest = experimental / f"{track}_lead_vocals.flac"
        backing_dest = experimental / f"{track}_backing_vocals.flac"
        _write_flac(lead_dest, best_lead, vocal_sr)
        _write_flac(backing_dest, best_backing, vocal_sr)

        fast_rebuilt = fast_lead + best_backing
        fast_residual = fast_parent - fast_rebuilt
        fast_parent_rms = float(np.sqrt(np.mean(fast_parent * fast_parent) + 1e-12))
        fast_residual_rms = float(np.sqrt(np.mean(fast_residual * fast_residual) + 1e-12))

        timings["vocal_alt_parent"] = round(alt_elapsed, 3)
        timings["vocal_quality_karaoke"] = round(quality_elapsed, 3)
        timings["vocal_fast_karaoke"] = round(fast_elapsed, 3)
        timings["karaoke_bs_roformer"] = round(alt_elapsed + quality_elapsed + fast_elapsed, 3)
        vocal_files = [lead_dest.name, backing_dest.name]
        vocal_report = {
            "ok": True,
            "benchmark_id": "vocal_benchmark_v1",
            "files": vocal_files,
            "parent_recipe": "BS-RoFormer-SW vocals",
            "lead_recipe": "25% SW + 75% MelBand Becruily vocal parent -> Becruily karaoke secondary/Instrumental",
            "backing_recipe": "SW vocals - fast SW->Becruily karaoke secondary/Instrumental",
            "best_stems_share_single_parent_pair": False,
            "fast_backing_route_parent_reconstruction_cosine": round(float(_cos(fast_parent, fast_rebuilt)), 9),
            "fast_backing_route_residual_relative_to_parent_db": _db(fast_residual_rms / max(fast_parent_rms, 1e-12)),
        }

'''

new = '''        lead_n = min(len(quality_lead), blend_n)
        best_lead = np.asarray(quality_lead[:lead_n], dtype=np.float32)

        def _vx_scaled_duplicate_analysis(lead_audio, backing_audio, sr):
            """Detect a backing stem that is mostly a gain-scaled copy of lead."""
            lead_arr = np.asarray(lead_audio, dtype=np.float32)
            backing_arr = np.asarray(backing_audio, dtype=np.float32)
            if lead_arr.ndim > 1:
                lead_mono = np.mean(lead_arr, axis=1)
            else:
                lead_mono = lead_arr
            if backing_arr.ndim > 1:
                backing_mono = np.mean(backing_arr, axis=1)
            else:
                backing_mono = backing_arr

            n = min(len(lead_mono), len(backing_mono))
            window = max(1, int(sr * 3.0))
            active_windows = 0
            duplicate_windows = 0
            duplicate_gain_db = []
            duplicate_corr = []
            duplicate_fit_db = []

            for start_idx in range(0, max(0, n - window + 1), window):
                lead_win = np.asarray(lead_mono[start_idx:start_idx + window], dtype=np.float64)
                backing_win = np.asarray(backing_mono[start_idx:start_idx + window], dtype=np.float64)
                lead_rms = float(np.sqrt(np.mean(lead_win * lead_win) + 1e-12))
                backing_rms = float(np.sqrt(np.mean(backing_win * backing_win) + 1e-12))
                lead_db = _db(lead_rms)
                backing_db = _db(backing_rms)
                if max(lead_db, backing_db) < -55.0 or min(lead_db, backing_db) < -60.0:
                    continue

                active_windows += 1
                denom = float(np.linalg.norm(lead_win) * np.linalg.norm(backing_win))
                corr = abs(float(np.dot(lead_win, backing_win) / denom)) if denom > 1e-12 else 0.0
                lead_energy = float(np.dot(lead_win, lead_win))
                scale = float(np.dot(lead_win, backing_win) / max(lead_energy, 1e-12))
                fitted = backing_win - (scale * lead_win)
                fitted_rms = float(np.sqrt(np.mean(fitted * fitted) + 1e-12))
                fitted_relative_db = _db(fitted_rms / max(backing_rms, 1e-12))
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
                p10, p90 = np.percentile(np.asarray(duplicate_gain_db, dtype=np.float64), [10, 90])
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
                "median_gain_db": round(median_gain_db, 3) if median_gain_db is not None else None,
                "gain_spread_db_p10_p90": round(gain_spread_db, 3) if gain_spread_db is not None else None,
                "median_correlation": round(median_corr, 6) if median_corr is not None else None,
                "median_scaled_fit_residual_db": round(median_fit_db, 3) if median_fit_db is not None else None,
                "thresholds": {
                    "window_seconds": 3.0,
                    "correlation": 0.985,
                    "scaled_fit_residual_db": -18.0,
                    "duplicate_fraction": 0.55,
                    "gain_spread_db_p10_p90": 3.0,
                },
                "backing_detected": not suppress,
                "suppressed_as_gain_scaled_duplicate": suppress,
            }

        # Live Experimental validation showed the two public vocal labels were
        # reversed relative to what users hear. Keep the benchmark routes intact
        # internally, but export them under the corrected public roles.
        public_lead = best_backing
        public_backing = best_lead
        duplicate_analysis = _vx_scaled_duplicate_analysis(public_lead, public_backing, vocal_sr)

        lead_dest = experimental / f"{track}_lead_vocals.flac"
        backing_dest = experimental / f"{track}_backing_vocals.flac"
        _write_flac(lead_dest, public_lead, vocal_sr)
        vocal_files = [lead_dest.name]
        if duplicate_analysis["backing_detected"]:
            _write_flac(backing_dest, public_backing, vocal_sr)
            vocal_files.append(backing_dest.name)

        fast_rebuilt = fast_lead + best_backing
        fast_residual = fast_parent - fast_rebuilt
        fast_parent_rms = float(np.sqrt(np.mean(fast_parent * fast_parent) + 1e-12))
        fast_residual_rms = float(np.sqrt(np.mean(fast_residual * fast_residual) + 1e-12))

        timings["vocal_alt_parent"] = round(alt_elapsed, 3)
        timings["vocal_quality_karaoke"] = round(quality_elapsed, 3)
        timings["vocal_fast_karaoke"] = round(fast_elapsed, 3)
        timings["karaoke_bs_roformer"] = round(alt_elapsed + quality_elapsed + fast_elapsed, 3)
        vocal_report = {
            "ok": True,
            "benchmark_id": "vocal_benchmark_v1",
            "files": vocal_files,
            "parent_recipe": "BS-RoFormer-SW vocals",
            "lead_recipe": "corrected public role: SW vocals - fast SW->Becruily karaoke secondary/Instrumental",
            "backing_recipe": "corrected public role: 25% SW + 75% MelBand Becruily vocal parent -> Becruily karaoke secondary/Instrumental",
            "public_role_correction": "live_validation_swap_v1",
            "backing_detected": bool(duplicate_analysis["backing_detected"]),
            "backing_suppressed_reason": (
                None if duplicate_analysis["backing_detected"]
                else "gain_scaled_duplicate_of_lead"
            ),
            "backing_duplicate_analysis": duplicate_analysis,
            "best_stems_share_single_parent_pair": False,
            "fast_backing_route_parent_reconstruction_cosine": round(float(_cos(fast_parent, fast_rebuilt)), 9),
            "fast_backing_route_residual_relative_to_parent_db": _db(fast_residual_rms / max(fast_parent_rms, 1e-12)),
        }

'''

if 'suppressed_as_gain_scaled_duplicate' not in text:
    if old not in text:
        raise RuntimeError('Could not locate locked vocal export block for duplicate guard')
    text = text.replace(old, new, 1)

# Keep QA model names honest about the production recipes.
text = text.replace(
    'label, model = "lead_vocals", "BS-RoFormer Karaoke"',
    'label, model = "lead_vocals", "Becruily Karaoke (25% SW + 75% MelBand parent)"',
)
text = text.replace(
    'label, model = "backing_vocals", "BS-RoFormer Karaoke (parent-minus-lead)"',
    'label, model = "backing_vocals", "Becruily Karaoke (SW parent-minus-fast-lead)"',
)

exp_path.write_text(text, encoding='utf-8')

check = exp_path.read_text(encoding='utf-8')
assert 'suppressed_as_gain_scaled_duplicate' in check
assert '"backing_detected": bool(duplicate_analysis["backing_detected"])' in check
assert 'gain_scaled_duplicate_of_lead' in check
print('LiteLABS gain-scaled duplicate backing-vocal guard applied')
