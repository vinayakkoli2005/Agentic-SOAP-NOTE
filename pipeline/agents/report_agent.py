import re
import json
import zipfile
from datetime import datetime


def _hr(char="=", n=78):
    return char * n


def _normalize_offset(offset_value):
    if offset_value is None:
        return None
    if isinstance(offset_value, (list, tuple)) and len(offset_value) >= 2:
        return (offset_value[0], offset_value[1])
    if isinstance(offset_value, dict):
        start_value = offset_value.get("start")
        end_value   = offset_value.get("end")
        if start_value is not None and end_value is not None:
            return (start_value, end_value)
    return None


def _format_offset(offset_value):
    normalized = _normalize_offset(offset_value)
    if normalized is None:
        return ""
    return f" [{normalized[0]}:{normalized[1]}]"


def _safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _get_rouge_metrics(eval_metrics):
    per_section = eval_metrics.get("per_section", {})
    macro = per_section.get("macro") if isinstance(per_section, dict) else None
    if macro:
        return {
            "rouge1": float(macro.get("rouge1_f1", 0) or 0),
            "rouge2": 0.0,
            "rougeL": float(macro.get("rougeL_f1", 0) or 0),
        }
    rouge_block = eval_metrics.get("rouge", {}) or eval_metrics.get("rouge_scores", {}) or {}
    rouge1 = (
        _safe_get(rouge_block, "rouge1", "f1") or
        _safe_get(rouge_block, "rouge-1", "f1") or
        rouge_block.get("rouge1") or rouge_block.get("rouge-1") or 0
    )
    rouge2 = (
        _safe_get(rouge_block, "rouge2", "f1") or
        _safe_get(rouge_block, "rouge-2", "f1") or
        rouge_block.get("rouge2") or rouge_block.get("rouge-2") or 0
    )
    rougeL = (
        _safe_get(rouge_block, "rougeL", "f1") or
        _safe_get(rouge_block, "rouge-l", "f1") or
        _safe_get(rouge_block, "rouge_l", "f1") or
        rouge_block.get("rougeL") or rouge_block.get("rouge-l") or rouge_block.get("rouge_l") or 0
    )
    return {"rouge1": float(rouge1 or 0), "rouge2": float(rouge2 or 0), "rougeL": float(rougeL or 0)}


def _get_bleu_metric(eval_metrics):
    per_section = eval_metrics.get("per_section", {})
    macro = per_section.get("macro") if isinstance(per_section, dict) else None
    if macro:
        return float(macro.get("bleu", 0) or 0)
    bleu_block = eval_metrics.get("bleu", {}) or {}
    if isinstance(bleu_block, dict):
        bleu_value = (
            bleu_block.get("score") or bleu_block.get("bleu") or bleu_block.get("value") or 0
        )
    else:
        bleu_value = bleu_block or 0
    return float(bleu_value or 0)


def _format_ner_entities(entities):
    lines = []
    for idx, e in enumerate(entities, 1):
        text_value   = e.get("text", "")
        label_value  = e.get("label", "")
        status_value = e.get("status", "")
        offset_str   = _format_offset(e.get("first_offset"))
        normalized   = e.get("normalized")
        snomed_val   = e.get("snomed")
        icd_val      = e.get("icd10_cm")
        left_part = f'{idx:>3}. [{label_value:<10} | {status_value:<14}] "{text_value}"{offset_str}'
        extra_parts = []
        if normalized and normalized != text_value:
            extra_parts.append(f'norm: "{normalized}"')
        if snomed_val:
            extra_parts.append(f'SNOMED: {snomed_val.get("concept_id")} | {snomed_val.get("term")}')
        if icd_val and icd_val.get("code"):
            extra_parts.append(f'ICD-10-CM: {icd_val.get("code")} | {icd_val.get("description")}')
        if extra_parts:
            left_part += " | " + " | ".join(extra_parts)
        lines.append(left_part)
    return "\n".join(lines) if lines else "No entities extracted"


def _format_keywords_for_ksoap(entities):
    lines = []
    for idx, e in enumerate(entities, 1):
        span   = e.get("text", "")
        norm   = e.get("normalized") or span
        label  = e.get("label", "Unknown")
        status = e.get("status", "Unknown")
        lines.append(f"{idx:>3}. ({span}, {norm}) [{label}, {status}]")
    return "\n".join(lines) if lines else "None documented"


def _format_snomed(entities):
    lines = []
    count = 1
    for e in entities:
        sn = e.get("snomed")
        if not sn:
            continue
        src_text = e.get("text", "")
        src_norm = e.get("normalized") or src_text
        icd      = e.get("icd10_cm")
        line     = f'{count:>3}. {sn.get("concept_id")}  {sn.get("term")}  <- "{src_norm}"'
        if icd and icd.get("code"):
            line += f'  -> {icd.get("code")}'
        lines.append(line)
        count += 1
    return "\n".join(lines) if lines else "No SNOMED mappings"


def _format_icd(entities):
    patient_lines = []
    family_lines  = []
    for e in entities:
        icd = e.get("icd10_cm")
        if not icd or not icd.get("code"):
            continue
        text_value   = e.get("text", "")
        status_value = e.get("status", "")
        offset_str   = _format_offset(e.get("first_offset"))
        line = f'- "{text_value}"{offset_str} -> {icd.get("code")} ({icd.get("description")})'
        if status_value == "Family_History":
            family_lines.append(line)
        elif status_value != "Negated":
            patient_lines.append(line)
    lines = []
    lines.append("Patient Diagnoses/Symptoms:")
    lines.append("\n".join(patient_lines) if patient_lines else "None")
    lines.append("")
    lines.append("Family History (NOT patient diagnoses):")
    lines.append("\n".join(family_lines) if family_lines else "None")
    return "\n".join(lines)


def _format_cpt(cpt_codes):
    if not cpt_codes:
        return "(No procedures explicitly ordered in transcript)"
    lines = []
    for idx, c in enumerate(cpt_codes, 1):
        if isinstance(c, dict):
            lines.append(f"{idx:>3}. {c.get('code', '')} | {c.get('description', '')}")
        else:
            lines.append(f"{idx:>3}. {str(c)}")
    return "\n".join(lines)


def _format_soap(soap_obj, soap_raw=""):
    if isinstance(soap_obj, dict) and soap_obj:
        sections = []
        for key in ["K", "S", "O", "A", "P"]:
            if key in soap_obj and soap_obj[key]:
                sections.append(f"{key}: {soap_obj[key]}")
        return "\n\n".join(sections) if sections else (soap_raw or "No SOAP generated")
    return soap_raw or "No SOAP generated"


def _format_ai_suggestion(soap_obj, soap_raw=""):
    if isinstance(soap_obj, dict):
        ai_text = (
            soap_obj.get("AI Suggestion") or
            soap_obj.get("AI_Suggestion") or
            soap_obj.get("AI")
        )
        if ai_text:
            return f"[Disclaimer]\n{ai_text}"
    return "[Disclaimer]\nNo AI suggestion generated"


def _format_eval(eval_metrics):
    kc     = eval_metrics.get("keyword_coverage", {})
    hall   = eval_metrics.get("hallucination", {})
    judge  = eval_metrics.get("llm_judge", {})
    cons   = eval_metrics.get("consistency", {})
    rouge_vals = _get_rouge_metrics(eval_metrics)
    bleu_val   = _get_bleu_metric(eval_metrics)
    lines = []
    lines.append(f"Keyword Coverage: {kc.get('coverage', 0):.1%} ({kc.get('found', 0)}/{kc.get('total', 0)})")
    if kc.get("missing"):
        lines.append(f"Missing Keywords: {', '.join(map(str, kc.get('missing', [])))}")
    lines.append(f"Hallucinations: {hall.get('count', 0)}")
    if hall.get("items"):
        lines.append("Hallucination Items:")
        for item in hall.get("items", []):
            lines.append(f"  - {item}")
    lines.append(f"Entity-SOAP Consistency: {cons.get('score', 0):.1%}")
    if cons.get("issues"):
        lines.append("Consistency Issues:")
        for item in cons.get("issues", []):
            lines.append(f"  - {item}")
    lines.append(
        f"ROUGE: R-1={rouge_vals['rouge1']:.4f} | "
        f"R-2={rouge_vals['rouge2']:.4f} | "
        f"R-L={rouge_vals['rougeL']:.4f}"
    )
    lines.append(f"BLEU: {bleu_val:.4f}")
    lines.append(f"LLM Judge Composite: {judge.get('composite', 0):.2f}")
    if judge:
        lines.append(
            f"Judge Breakdown: Completeness={judge.get('completeness', 0)} | "
            f"Correctness={judge.get('correctness', 0)} | "
            f"Coherence={judge.get('coherence', 0)} | "
            f"Assessment={judge.get('assessment_quality', judge.get('assessment', 0))} | "
            f"Plan Safety={judge.get('plan_safety', judge.get('plan', 0))}"
        )
    return "\n".join(lines)


class ReportAgent:
    """Builds, saves, and exports the comprehensive pipeline report."""

    def run(self, all_results: dict, raw_transcript: str, diarized_transcript: str,
            t_asr_elapsed: float = 0.0, t_diar_elapsed: float = 0.0,
            output_dir: str = ".") -> dict:
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        t_total = round(
            t_asr_elapsed + t_diar_elapsed +
            sum(r.get("time_s", 0) for r in all_results.values()),
            2,
        )

        report_lines = []
        report_lines.append(_hr())
        report_lines.append("COMPREHENSIVE CLINICAL NLP PIPELINE REPORT")
        report_lines.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
        report_lines.append(
            f"Total Time: {t_total}s (ASR={t_asr_elapsed}s | Diarization={t_diar_elapsed}s)"
        )
        report_lines.append(_hr())

        report_lines.append("\nRAW ASR TRANSCRIPT")
        report_lines.append("-" * 78)
        report_lines.append(raw_transcript or "No ASR data")

        report_lines.append("\nDIARIZED / PROCESSED TRANSCRIPT")
        report_lines.append("-" * 78)
        report_lines.append(diarized_transcript or "No diarization")

        report_lines.append("\nNER OUTPUT BY CONFIG")
        report_lines.append("-" * 78)
        for name, res in all_results.items():
            report_lines.append(f"\n[{name}]")
            report_lines.append(_format_ner_entities(res.get("entities", [])))

        report_lines.append("\nK-SOAP KEYWORDS BY CONFIG")
        report_lines.append("-" * 78)
        for name, res in all_results.items():
            report_lines.append(f"\n[{name}]")
            report_lines.append(_format_keywords_for_ksoap(res.get("entities", [])))

        report_lines.append("\nSOAP NOTES BY CONFIG")
        report_lines.append("-" * 78)
        for name, res in all_results.items():
            report_lines.append(f"\n[{name}]")
            report_lines.append(_format_soap(res.get("soap", {}), res.get("soap_raw", "")))

        report_lines.append("\nAI SUGGESTIONS BY CONFIG")
        report_lines.append("-" * 78)
        for name, res in all_results.items():
            report_lines.append(f"\n[{name}]")
            report_lines.append(_format_ai_suggestion(res.get("soap", {}), res.get("soap_raw", "")))

        report_lines.append("\nSNOMED CT MAPPINGS BY CONFIG")
        report_lines.append("-" * 78)
        for name, res in all_results.items():
            report_lines.append(f"\n[{name}]")
            report_lines.append(_format_snomed(res.get("entities", [])))

        report_lines.append("\nICD-10-CM CODES BY CONFIG")
        report_lines.append("-" * 78)
        for name, res in all_results.items():
            report_lines.append(f"\n[{name}]")
            report_lines.append(_format_icd(res.get("entities", [])))

        report_lines.append("\nCPT CODES BY CONFIG")
        report_lines.append("-" * 78)
        for name, res in all_results.items():
            report_lines.append(f"\n[{name}]")
            report_lines.append(_format_cpt(res.get("cpt_codes", [])))

        report_lines.append("\nTIMING BY CONFIG")
        report_lines.append("-" * 78)
        for name, res in all_results.items():
            report_lines.append(f"{name}: {res.get('time_s', 0)}s")

        report_lines.append("\nEVALUATION BY CONFIG")
        report_lines.append("-" * 78)
        for name, res in all_results.items():
            report_lines.append(f"\n[{name}]")
            report_lines.append(_format_eval(res.get("eval_metrics", {})))

        # Comparison summary table
        report_lines.append(f"\n{_hr()}")
        report_lines.append("COMPARISON SUMMARY")
        report_lines.append(_hr())
        header = (
            f"{'Config':<25} {'KW Cov':>8} {'Consist':>9} "
            f"{'Halluc':>8} {'R-L':>8} {'BLEU':>8} {'Judge':>8} {'Time':>8}"
        )
        report_lines.append(header)
        report_lines.append("-" * len(header))
        for name, res in all_results.items():
            em         = res.get("eval_metrics", {})
            kc         = em.get("keyword_coverage", {})
            hall       = em.get("hallucination", {})
            judge      = em.get("llm_judge", {})
            cons       = em.get("consistency", {})
            rouge_vals = _get_rouge_metrics(em)
            bleu_val   = _get_bleu_metric(em)
            line = (
                f"{name:<25} "
                f"{kc.get('coverage', 0):>8.1%} "
                f"{cons.get('score', 0):>9.1%} "
                f"{hall.get('count', 0):>8} "
                f"{rouge_vals['rougeL']:>8.4f} "
                f"{bleu_val:>8.4f} "
                f"{judge.get('composite', 0):>8.2f} "
                f"{res.get('time_s', 0):>7.0f}s"
            )
            report_lines.append(line)

        report_txt = "\n".join(report_lines)

        # Write TXT
        import os
        report_path = os.path.join(output_dir, f"Clinical_Report_{ts}.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_txt)

        # Build JSON export
        json_export = {
            "generated": datetime.now().isoformat(),
            "timing": {
                "total_s":        t_total,
                "asr_s":          t_asr_elapsed,
                "diarization_s":  t_diar_elapsed,
            },
            "raw_transcript":      raw_transcript or "",
            "diarized_transcript": diarized_transcript or "",
            "results": {},
        }
        for name, res in all_results.items():
            em = res.get("eval_metrics", {})
            json_export["results"][name] = {
                "config":    res.get("config", {}),
                "time_s":    res.get("time_s", 0),
                "ksoap_keywords": [
                    {
                        "span":      e.get("text"),
                        "entity":    e.get("normalized") or e.get("text"),
                        "label":     e.get("label"),
                        "assertion": e.get("status"),
                        "offset":    _normalize_offset(e.get("first_offset")),
                    }
                    for e in res.get("entities", [])
                ],
                "soap":     res.get("soap", {}),
                "soap_raw": res.get("soap_raw", ""),
                "ai_suggestion": (
                    _safe_get(res, "soap", "AI Suggestion") or
                    _safe_get(res, "soap", "AI_Suggestion") or ""
                ),
                "entities": [
                    {
                        "text":       e.get("text"),
                        "label":      e.get("label"),
                        "status":     e.get("status"),
                        "offset":     _normalize_offset(e.get("first_offset")),
                        "normalized": e.get("normalized"),
                        "snomed": {
                            "id":   e["snomed"]["concept_id"],
                            "term": e["snomed"]["term"],
                        } if e.get("snomed") else None,
                        "icd10_cm": {
                            "code":        e["icd10_cm"]["code"],
                            "description": e["icd10_cm"]["description"],
                        } if e.get("icd10_cm") and e["icd10_cm"].get("code") else None,
                    }
                    for e in res.get("entities", [])
                ],
                "cpt_codes":   res.get("cpt_codes", []),
                "eval_metrics": em,
                "objective_metrics": {
                    "rouge": _get_rouge_metrics(em),
                    "bleu":  _get_bleu_metric(em),
                },
            }

        json_path = os.path.join(output_dir, f"Clinical_Result_{ts}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_export, f, indent=2, default=str)

        # Zip both
        zip_path = os.path.join(output_dir, f"Clinical_Output_{ts}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(report_path, arcname=f"Clinical_Report_{ts}.txt")
            zipf.write(json_path,   arcname=f"Clinical_Result_{ts}.json")

        print(f"\n{'#'*65}")
        print("EXPORT COMPLETE")
        print(f"TXT : {report_path}")
        print(f"JSON: {json_path}")
        print(f"ZIP : {zip_path}")
        print(f"{'#'*65}")

        return {
            "report_path": report_path,
            "json_path":   json_path,
            "zip_path":    zip_path,
            "report_txt":  report_txt,
        }
