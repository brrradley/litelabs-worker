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

        # The karaoke run already produced both outputs. Keep the model's direct
        # primary/Vocals output as a backing candidate instead of assuming the
        # arithmetic parent-minus-secondary residual is always backing.
        direct_backing_path = _vx_find_output(fast_out, "vocals")
        direct_backing = None
        if direct_backing_path is not None:
            direct_backing, _ = _read(direct_backing_path)
            direct_n = min(len(direct_backing), len(fast_parent))
            direct_backing = np.asarray(direct_backing[:direct_n], dtype=np.float32)

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

        # Do not trust separator output names to define vocal roles. Live tests
        # showed the nominal karaoke Instrumental/secondary output can be either
        # isolated vocal content or accompaniment depending on the source.
        def _vx_role_evidence(candidate_audio, vocal_parent_audio, mixture_audio):
            candidate = np.asarray(candidate_audio, dtype=np.float32)
            vocal_parent_arr = np.asarray(vocal_parent_audio, dtype=np.float32)
            mixture_arr = np.asarray(mixture_audio, dtype=np.float32)
            n_role = min(len(candidate), len(vocal_parent_arr), len(mixture_arr))
            if n_role <= 0:
                return {
                    "vocal_cosine": 0.0,
                    "instrumental_cosine": 0.0,
                    "vocal_margin": -1.0,
                    "vocal_like": False,
                }

            candidate = candidate[:n_role]
            vocal_parent_arr = vocal_parent_arr[:n_role]
            mixture_arr = mixture_arr[:n_role]
            instrumental_ref = mixture_arr - vocal_parent_arr

            def _flat_mono(item):
                arr = np.asarray(item, dtype=np.float32)
                if arr.ndim > 1:
                    arr = np.mean(arr, axis=1)
                return np.asarray(arr, dtype=np.float32).reshape(-1)

            candidate_mono = _flat_mono(candidate)
            vocal_mono = _flat_mono(vocal_parent_arr)
            instrumental_mono = _flat_mono(instrumental_ref)
            vocal_cosine = abs(float(_cos(candidate_mono, vocal_mono)))
            instrumental_cosine = abs(float(_cos(candidate_mono, instrumental_mono)))
            margin = vocal_cosine - instrumental_cosine
            vocal_like = bool(
                vocal_cosine >= 0.16
                and margin >= 0.035
            )
            return {
                "vocal_cosine": round(vocal_cosine, 6),
                "instrumental_cosine": round(instrumental_cosine, 6),
                "vocal_margin": round(margin, 6),
                "vocal_like": vocal_like,
            }

        quality_evidence = _vx_role_evidence(best_lead, sw_vocals_audio, mixture)
        residual_evidence = _vx_role_evidence(best_backing, sw_vocals_audio, mixture)
        direct_backing_evidence = (
            _vx_role_evidence(direct_backing, sw_vocals_audio, mixture)
            if direct_backing is not None
            else {
                "vocal_cosine": 0.0,
                "instrumental_cosine": 1.0,
                "vocal_margin": -1.0,
                "vocal_like": False,
            }
        )
        fast_secondary_evidence = _vx_role_evidence(fast_lead, sw_vocals_audio, mixture)

        # live_validated_vocal_roles_v3
        # Live auditioning established that the quality karaoke output can be a
        # clean backing-vocal subset while a separate candidate can still carry
        # the entire vocal parent. Do not export an "all vocals" file as backing.
        #
        # Use the trusted SW vocal parent as the mass-conserving vocal reference:
        #   backing = validated backing subset
        #   lead    = SW vocal parent - backing
        #
        # This guarantees lead + backing reconstruct the vocal parent and makes
        # the public labels independent of ambiguous model output filenames.
        parent_vocals = np.asarray(fast_parent, dtype=np.float32)
        backing_raw = np.asarray(best_lead, dtype=np.float32)
        vocal_n = min(len(parent_vocals), len(backing_raw))
        parent_vocals = parent_vocals[:vocal_n]
        backing_raw = backing_raw[:vocal_n]

        parent_flat = np.asarray(parent_vocals, dtype=np.float64).reshape(-1)
        backing_flat = np.asarray(backing_raw, dtype=np.float64).reshape(-1)
        parent_rms = float(np.sqrt(np.mean(parent_flat * parent_flat) + 1e-12))
        backing_rms = float(np.sqrt(np.mean(backing_flat * backing_flat) + 1e-12))
        backing_to_parent_ratio = backing_rms / max(parent_rms, 1e-12)
        backing_parent_cosine = abs(float(_cos(backing_raw, parent_vocals)))

        # A useful backing subset should be meaningfully smaller than the entire
        # vocal parent while still clearly belonging to it. If it fails this
        # check, omit backing rather than exporting an all-vocal impostor.
        backing_subset_valid = bool(
            vocal_n > 0
            and backing_to_parent_ratio >= 0.025
            and backing_to_parent_ratio <= 0.88
            and backing_parent_cosine >= 0.10
        )

        backing_gain = 1.0
        if backing_subset_valid:
            denom = float(np.dot(backing_flat, backing_flat))
            if denom > 1e-12:
                fitted_gain = float(np.dot(parent_flat, backing_flat) / denom)
                if np.isfinite(fitted_gain):
                    backing_gain = float(np.clip(fitted_gain, 0.65, 1.35))

        public_backing = np.asarray(backing_raw * backing_gain, dtype=np.float32)
        public_lead = np.asarray(parent_vocals - public_backing, dtype=np.float32)
        public_lead_route = "sw_parent_minus_live_validated_backing"
        backing_route = "quality_karaoke_live_validated_backing"

        duplicate_analysis = (
            _vx_scaled_duplicate_analysis(public_lead, public_backing, vocal_sr)
            if backing_subset_valid
            else {
                "backing_detected": False,
                "suppressed_as_gain_scaled_duplicate": False,
                "reason": "quality_candidate_not_valid_backing_subset",
            }
        )

        lead_dest = experimental / f"{track}_lead_vocals.flac"
        backing_dest = experimental / f"{track}_backing_vocals.flac"
        _write_flac(lead_dest, public_lead, vocal_sr)
        vocal_files = [lead_dest.name]

        backing_exported = bool(
            backing_subset_valid
            and duplicate_analysis.get("backing_detected")
        )
        if backing_exported:
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
            "lead_recipe": public_lead_route,
            "backing_recipe": backing_route,
            "public_role_correction": "live_validated_vocal_roles_v3",
            "vocal_mass_conservation": "lead_plus_backing_equals_sw_vocal_parent",
            "backing_subset_validation": {
                "valid": bool(backing_subset_valid),
                "rms_ratio_to_parent": round(float(backing_to_parent_ratio), 6),
                "parent_cosine": round(float(backing_parent_cosine), 6),
                "fitted_gain": round(float(backing_gain), 6),
            },
            "vocal_role_evidence": {
                "quality_backing_candidate": quality_evidence,
                "fast_secondary": fast_secondary_evidence,
                "direct_primary": direct_backing_evidence,
                "legacy_residual": residual_evidence,
            },
            "backing_detected": bool(backing_exported),
            "backing_suppressed_reason": (
                None if backing_exported
                else (
                    "gain_scaled_duplicate_of_lead"
                    if backing_is_vocal and duplicate_analysis.get("suppressed_as_gain_scaled_duplicate")
                    else "rejected_as_non_vocal_or_ambiguous"
                )
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
assert 'live_validated_vocal_roles_v3' in check
assert 'lead_plus_backing_equals_sw_vocal_parent' in check
assert '_vx_role_evidence' in check
assert '"backing_detected": bool(backing_exported)' in check
assert 'gain_scaled_duplicate_of_lead' in check
print('LiteLABS gain-scaled duplicate backing-vocal guard applied')
