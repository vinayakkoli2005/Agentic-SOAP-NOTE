"""
SOAP Note Generation Pipeline — Agentic Entry Point
Exact replication of soap-note-generation.ipynb, one agent per task.
"""

import os
import time
import torch
from concurrent.futures import ThreadPoolExecutor

import pipeline.models as models
from pipeline.agents.asr_agent          import ASRAgent
from pipeline.agents.diarization_agent  import DiarizationAgent
from pipeline.agents.ner_agent          import NERAgent
from pipeline.agents.code_enrichment_agent import CodeEnrichmentAgent
from pipeline.agents.soap_agent         import SOAPAgent
from pipeline.agents.evaluation_agent   import EvaluationAgent
from pipeline.agents.report_agent       import ReportAgent
from pipeline.utils                     import verify_code_llm
from pipeline.config                    import USE_CODE_VERIFICATION


# ── Clinical inference (exact from notebook Cell 16) ─────────────────────────

def _get_confirmed_data(coded_entities):
    ids, texts = set(), set()
    for e in coded_entities:
        if e["status"] == "Confirmed":
            texts.add(e["text"].lower())
            sn = e.get("snomed")
            if sn:
                ids.add(sn["concept_id"])
    return ids, texts

def _has(sn_ids, texts, target_ids, target_kw):
    return (
        any(t in sn_ids for t in target_ids) or
        any(kw in txt for txt in texts for kw in target_kw)
    )

def _make_inferred(text, sn_id, sn_term, icd_code, icd_desc):
    return {
        "text": text, "label": "Disease", "status": "Confirmed",
        "reasoning": "Deterministic clinical inference",
        "first_offset": None, "_inferred": True,
        "snomed": {
            "concept_id": sn_id, "term": sn_term,
            "entity_text": text, "method": "clinical_inference",
            "crosswalk_icd10": icd_code, "crosswalk_icd10_desc": icd_desc,
        },
        "icd10_cm": {
            "code": icd_code, "description": icd_desc,
            "cosine_score": 1.0, "rerank_score": 10.0,
            "method": "clinical_inference",
        },
    }

def _run_clinical_inference(coded_entities):
    sn_ids, texts = _get_confirmed_data(coded_entities)
    inferred = []
    if (
        _has(sn_ids, texts, ["14094001"], ["missed period", "late period", "amenorrhea"]) and
        _has(sn_ids, texts, ["422587007", "422400008"], ["nausea", "vomiting", "nauseated"]) and
        _has(sn_ids, texts, ["162116003"], ["urinary frequency"])
    ):
        inferred.append(_make_inferred(
            "suspected pregnancy (inferred: amenorrhea + nausea + urinary frequency)",
            "77386006", "Pregnancy", "Z32.00",
            "Encounter for pregnancy test, result unknown",
        ))
    if (
        _has(sn_ids, texts, ["267036007"], ["shortness of breath", "difficulty breathing"]) and
        _has(sn_ids, texts, ["267038008"], ["swelling", "edema", "swollen"]) and
        _has(sn_ids, texts, [], ["orthopnea", "lying flat", "pillows", "crackles", "heart failure"])
    ):
        inferred.append(_make_inferred(
            "CHF exacerbation (inferred: dyspnea + edema + orthopnea)",
            "42343007", "Congestive heart failure", "I50.9", "Heart failure, unspecified",
        ))
    if (
        _has(sn_ids, texts, ["29857009"], ["chest pain", "chest pressure"]) and
        _has(sn_ids, texts, ["267036007"], ["shortness of breath", "difficulty breathing"]) and
        _has(sn_ids, texts, [], ["smoking", "smoker", "cocaine", "high blood pressure",
                                  "hypertension", "diabetes"])
    ):
        inferred.append(_make_inferred(
            "suspected ACS (inferred: chest pain + dyspnea + risk factors)",
            "394659003", "Acute coronary syndrome", "I24.9",
            "Acute ischemic heart disease, unspecified",
        ))
    if (
        _has(sn_ids, texts, ["62315008"], ["diarrhea", "loose stools"]) and
        _has(sn_ids, texts, ["422587007", "422400008"], ["nausea", "vomiting", "nauseated"]) and
        _has(sn_ids, texts, ["21522001", "73063007"], ["abdominal pain", "belly pain",
                                                         "cramp", "crampy"])
    ):
        inferred.append(_make_inferred(
            "suspected gastroenteritis (inferred: diarrhea + N/V + abdominal pain)",
            "235856003", "Gastroenteritis", "K52.9",
            "Noninfective gastroenteritis and colitis, unspecified",
        ))
    return inferred


# ── Single-config pipeline runner ─────────────────────────────────────────────

def run_full_pipeline(transcript_text, k_shot=3, strategy="query_aware",
                      reference_soap=None):
    t_start = time.time()

    print("\n--- Phase 1: NER (SPELL Architecture + VerifyNER) ---")
    ner_agent = NERAgent()
    entities, rejected = ner_agent.run(transcript_text)

    print("\n--- Phase 2: SOAP + Code Enrichment (parallel) ---")
    soap_agent = SOAPAgent()
    code_agent = CodeEnrichmentAgent()

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_soap  = pool.submit(soap_agent.run, transcript_text, entities, k_shot, strategy)
        f_codes = pool.submit(code_agent.run, entities, transcript_text)
        soap, soap_raw      = f_soap.result()
        entities_coded      = f_codes.result()

    if USE_CODE_VERIFICATION:
        print("\n--- Phase 2.25: Per-code LLM verification (MedCodER-style) ---")
        entities_coded = verify_code_llm(entities_coded, "snomed",   transcript_text)
        entities_coded = verify_code_llm(entities_coded, "icd10_cm", transcript_text)

    print("\n--- Phase 2.5: Clinical Inference (deterministic) ---")
    inferred_entities = _run_clinical_inference(entities_coded)
    for ie in inferred_entities:
        entities_coded.append(ie)
        print(f"  INFERRED: {ie['text']}")
    if not inferred_entities:
        print("  No inference patterns matched")

    print("\n--- Phase 3: CPT Assignment ---")
    cpt_codes = code_agent.assign_cpt(entities_coded)
    print(f"  CPT: {len(cpt_codes)} codes assigned")

    print("\n--- Phase 4: SOAP Quality Evaluation ---")
    eval_agent   = EvaluationAgent()
    eval_metrics = eval_agent.run(soap_raw, transcript_text, entities, reference_soap)

    if reference_soap:
        from pipeline.agents.evaluation_agent import evaluate_soap_per_section
        try:
            per_section = evaluate_soap_per_section(soap_raw, reference_soap)
            eval_metrics["per_section"] = per_section
            macro = per_section["macro"]
            print(
                f"         [Per-section vs reference] macro "
                f"ROUGE-L={macro['rougeL_f1']:.3f}  "
                f"BLEU={macro['bleu']:.3f}  "
                f"BERT={macro['bertscore_f1']:.3f}"
            )
        except Exception as e:
            print(f"         per-section eval failed: {e}")

    total        = time.time() - t_start
    snomed_count = sum(1 for e in entities_coded if e.get("snomed"))
    icd10_count  = sum(1 for e in entities_coded if e.get("icd10_cm") and e["icd10_cm"].get("code"))
    crosswalk_count = sum(
        1 for e in entities_coded
        if e.get("icd10_cm") and e["icd10_cm"].get("method") == "SNOMED_crosswalk"
    )
    j  = eval_metrics["llm_judge"]
    kc = eval_metrics["keyword_coverage"]
    print(f"\n{'='*65}")
    print(f"  PIPELINE COMPLETE  {total:.1f}s  ({k_shot}-shot, {strategy})")
    print(f"  Entities: {len(entities)} | Rejected: {len(rejected)}")
    print(f"  SNOMED: {snomed_count} | ICD-10: {icd10_count} ({crosswalk_count} crosswalk)")
    print(f"  CPT: {len(cpt_codes)}")
    print(
        f"  Quality: Judge={j['composite']}/5.0  "
        f"KW={kc['coverage']:.0%}  "
        f"Halluc={eval_metrics['hallucination']['count']}"
    )
    print(f"{'='*65}")

    return {
        "transcript":    transcript_text,
        "entities":      entities_coded,
        "rejected":      rejected,
        "soap":          soap,
        "soap_raw":      soap_raw,
        "cpt_codes":     cpt_codes,
        "eval_metrics":  eval_metrics,
        "config":        {"k_shot": k_shot, "strategy": strategy},
        "time_s":        round(total, 2),
    }


# ── Multi-config runner + export (mirrors notebook Cell 17) ──────────────────

CONFIGS = [
    {"name": "0-shot_v1_baseline",  "k_shot": 0, "strategy": "smallest"},
    {"name": "3-shot_smallest",     "k_shot": 3, "strategy": "smallest"},
    {"name": "3-shot_query_aware",  "k_shot": 3, "strategy": "query_aware"},
    {"name": "5-shot_smallest",     "k_shot": 5, "strategy": "smallest"},
]


def run_all_configs(transcript_text, reference_soap=None, output_dir="."):
    all_results = {}

    for cfg in CONFIGS:
        print(f"\n{'#'*70}")
        print(f"# RUNNING: {cfg['name']}")
        print(f"{'#'*70}")
        torch.cuda.empty_cache()
        try:
            result = run_full_pipeline(
                transcript_text,
                k_shot=cfg["k_shot"],
                strategy=cfg["strategy"],
                reference_soap=reference_soap,
            )
            all_results[cfg["name"]] = result
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  OOM: {cfg['name']} skipped (prompt too long for GPU)")
            all_results[cfg["name"]] = {
                "transcript":  transcript_text,
                "entities":    [], "rejected":    [],
                "soap":        {}, "soap_raw":    f"SKIPPED: OOM on {cfg['name']}",
                "cpt_codes":   [],
                "eval_metrics": {
                    "llm_judge":        {"composite": 0},
                    "keyword_coverage": {"coverage": 0, "found": 0, "total": 0, "missing": []},
                    "hallucination":    {"count": 0, "items": []},
                    "consistency":      {"score": 0, "issues": []},
                },
                "config":  {"k_shot": cfg["k_shot"], "strategy": cfg["strategy"]},
                "time_s":  0,
            }

    return all_results


def main(audio_path: str, reference_soap: str = None, output_dir: str = "."):
    print("=" * 70)
    print("CLINICAL NLP PIPELINE — AGENTIC RUNNER")
    print("=" * 70)

    print("\n[Loading models...]")
    models.load_all_models()

    # Phase 0: ASR
    print("\n--- Phase 0a: ASR ---")
    t_asr_start = time.time()
    asr_agent   = ASRAgent()
    raw_transcript = asr_agent.run(audio_path)
    t_asr_elapsed  = round(time.time() - t_asr_start, 2)

    # Phase 0b: Diarization
    print("\n--- Phase 0b: Diarization ---")
    t_diar_start    = time.time()
    diar_agent      = DiarizationAgent()
    diarized_transcript = diar_agent.run(raw_transcript)
    t_diar_elapsed  = round(time.time() - t_diar_start, 2)

    # Run all configs
    all_results = run_all_configs(diarized_transcript, reference_soap, output_dir)

    # Report + export
    report_agent = ReportAgent()
    paths = report_agent.run(
        all_results, raw_transcript, diarized_transcript,
        t_asr_elapsed, t_diar_elapsed, output_dir,
    )

    print(f"\nDone. Outputs written to: {output_dir}")
    return paths, all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SOAP Note Generation Pipeline")
    parser.add_argument("audio",           help="Path to input audio file")
    parser.add_argument("--reference",     help="Path to reference SOAP note (optional)", default=None)
    parser.add_argument("--output_dir",    help="Output directory",                       default=".")
    args = parser.parse_args()

    ref_soap = None
    if args.reference and os.path.isfile(args.reference):
        with open(args.reference, encoding="utf-8") as f:
            ref_soap = f.read()

    main(args.audio, ref_soap, args.output_dir)
