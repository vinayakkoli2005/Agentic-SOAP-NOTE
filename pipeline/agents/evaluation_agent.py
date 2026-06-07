import re
import math
import textwrap
from collections import Counter
from sentence_transformers import util

import pipeline.models as models
from pipeline.utils import _llm, _parse_json

# ── Tokenizer (exact from notebook Cell 15) ──────────────────────────────────

def _tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def rouge_n(candidate, reference, n=1):
    cand_tok = _tokenize(candidate)
    ref_tok  = _tokenize(reference)
    if not cand_tok or not ref_tok:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    cand_ng = Counter(tuple(cand_tok[i:i+n]) for i in range(len(cand_tok)-n+1))
    ref_ng  = Counter(tuple(ref_tok[i:i+n])  for i in range(len(ref_tok)-n+1))
    overlap = sum((cand_ng & ref_ng).values())
    p  = overlap / max(sum(cand_ng.values()), 1)
    r  = overlap / max(sum(ref_ng.values()),  1)
    f1 = 2 * p * r / max(p + r, 1e-8)
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}

def rouge_l(candidate, reference):
    cand_tok = _tokenize(candidate)
    ref_tok  = _tokenize(reference)
    if not cand_tok or not ref_tok:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    m, n = len(cand_tok), len(ref_tok)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if cand_tok[i-1] == ref_tok[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    p  = lcs / max(m, 1)
    r  = lcs / max(n, 1)
    f1 = 2 * p * r / max(p + r, 1e-8)
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}

def bleu_score(candidate, reference, max_n=4):
    cand_tok = _tokenize(candidate)
    ref_tok  = _tokenize(reference)
    if not cand_tok or not ref_tok:
        return {"bleu": 0.0, "bleu1": 0.0, "bleu2": 0.0, "bleu4": 0.0, "brevity_penalty": 0.0}
    c_len, r_len = len(cand_tok), len(ref_tok)
    bp = 1.0 if c_len >= r_len else math.exp(1 - r_len / c_len) if c_len > 0 else 0.0
    precisions  = []
    individual  = {}
    for n in range(1, max_n + 1):
        cand_ng  = Counter(tuple(cand_tok[i:i+n]) for i in range(max(len(cand_tok)-n+1, 0)))
        ref_ng   = Counter(tuple(ref_tok[i:i+n])  for i in range(max(len(ref_tok)-n+1, 0)))
        clipped  = sum(min(cand_ng[ng], ref_ng[ng]) for ng in cand_ng)
        total    = max(sum(cand_ng.values()), 1)
        p_n      = clipped / total
        precisions.append(p_n)
        individual[f"bleu{n}"] = round(p_n, 4)
    log_avg = 0.0
    weight  = 1.0 / max_n
    for p_n in precisions:
        if p_n == 0:
            log_avg = float('-inf')
            break
        log_avg += weight * math.log(p_n)
    bleu = 0.0 if log_avg == float('-inf') else bp * math.exp(log_avg)
    return {
        "bleu": round(bleu, 4),
        "bleu1": individual.get("bleu1", 0.0),
        "bleu2": individual.get("bleu2", 0.0),
        "bleu4": individual.get("bleu4", 0.0),
        "brevity_penalty": round(bp, 4),
    }

def bertscore_document(candidate, reference):
    if not candidate.strip() or not reference.strip():
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    c_emb = models.embedder.encode(candidate, convert_to_tensor=True)
    r_emb = models.embedder.encode(reference, convert_to_tensor=True)
    sim   = float(util.cos_sim(c_emb, r_emb)[0][0])
    return {"precision": round(sim, 4), "recall": round(sim, 4), "f1": round(sim, 4)}

def bertscore_sentence(candidate, reference):
    c_sents = [s.strip() for s in re.split(r'[.!?]+', candidate) if len(s.strip()) > 10]
    r_sents = [s.strip() for s in re.split(r'[.!?]+', reference) if len(s.strip()) > 10]
    if not c_sents or not r_sents:
        return bertscore_document(candidate, reference)
    c_embs     = models.embedder.encode(c_sents, convert_to_tensor=True)
    r_embs     = models.embedder.encode(r_sents, convert_to_tensor=True)
    sim_matrix = util.cos_sim(c_embs, r_embs)
    p  = float(sim_matrix.max(dim=1).values.mean())
    r  = float(sim_matrix.max(dim=0).values.mean())
    f1 = 2 * p * r / max(p + r, 1e-8)
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}

def keyword_coverage(soap_text, entities):
    if not entities:
        return {"coverage": 0.0, "found": 0, "total": 0, "missing": []}
    soap_lower = soap_text.lower()
    relevant   = [e for e in entities if e.get("status") in ("Confirmed", "Historical", "Family_History")]
    found, missing = 0, []
    for e in relevant:
        text  = e["text"].lower()
        words = [w for w in text.split() if len(w) >= 4]
        if text in soap_lower or any(w in soap_lower for w in words):
            found += 1
        else:
            missing.append(e["text"])
    total = len(relevant)
    return {"coverage": round(found / max(total, 1), 4), "found": found, "total": total, "missing": missing}

_KNOWN_DRUGS = re.compile(
    r'\b(?:metformin|lisinopril|amlodipine|aspirin|ibuprofen|acetaminophen|'
    r'amoxicillin|azithromycin|ciprofloxacin|prednisone|omeprazole|sumatriptan|'
    r'gabapentin|metoprolol|furosemide|atorvastatin|rosuvastatin|warfarin|'
    r'levothyroxine|losartan|albuterol|naproxen|hydrochlorothiazide|'
    r'clopidogrel|pantoprazole|sertraline|fluoxetine|diazepam|lorazepam)\b',
    re.I
)

def detect_hallucinations(soap_text, transcript, entities):
    hallucinated = []
    soap_lower   = soap_text.lower()
    trans_lower  = transcript.lower()

    soap_drugs  = set(m.group().lower() for m in _KNOWN_DRUGS.finditer(soap_text))
    trans_drugs = set(m.group().lower() for m in _KNOWN_DRUGS.finditer(transcript))
    ent_drugs   = set()
    for e in entities:
        if e.get("label") == "Drug":
            ent_drugs.update(m.group().lower() for m in _KNOWN_DRUGS.finditer(e["text"]))
    for drug in soap_drugs:
        if drug not in trans_drugs and drug not in ent_drugs:
            hallucinated.append(f"Drug not in transcript: {drug}")

    fabrication_checks = [
        (r'radiat\w+\s+to\s+(?:left\s+)?(?:arm|shoulder|jaw|back|neck)', "Radiating pain"),
        (r'doctor\s+ordered\s+(?:further\s+)?(?:evaluation|ecg|ekg|troponin|imaging|blood|labs)', "Doctor ordered tests"),
        (r'doctor\s+prescribed', "Doctor prescribed medication"),
    ]
    for pattern, label in fabrication_checks:
        if re.search(pattern, soap_lower):
            if not re.search(pattern, trans_lower):
                if "radiat" in pattern:
                    if re.search(r'radiat\w+\s+anywhere.*?\bno\b', trans_lower, re.DOTALL):
                        hallucinated.append(f"Fabricated: {label} (patient explicitly denied)")
                    elif "radiating anywhere" in trans_lower:
                        hallucinated.append(f"Fabricated: {label} (patient denied radiation)")
                elif "order" in pattern or "prescri" in pattern:
                    if not re.search(r"i'?m\s+(?:going\s+to\s+)?order", trans_lower):
                        if not re.search(r"i'?m\s+(?:going\s+to\s+)?start", trans_lower):
                            if not re.search(r"i'?m\s+prescribing", trans_lower):
                                hallucinated.append(f"Fabricated: {label} (not spoken by doctor)")
    return {"count": len(hallucinated), "items": hallucinated}

_JUDGE_PROMPT = textwrap.dedent("""
You are a clinical documentation quality evaluator.
Score the SOAP note against the transcript on these 5 dimensions (1-5 scale).

TRANSCRIPT:
{transcript}

SOAP NOTE:
{soap}

SCORING (1=poor, 5=excellent):
1. COMPLETENESS: All clinically relevant info captured?
2. CORRECTNESS: Everything grounded in transcript? No invented findings?
3. COHERENCE: Well-organized with standard medical terminology?
4. ASSESSMENT_QUALITY: Specific primary diagnosis with reasonable differentials?
5. PLAN_SAFETY: Plan reflects ONLY explicit doctor orders?

Return ONLY a JSON object:
{{"completeness": N, "correctness": N, "coherence": N, "assessment_quality": N, "plan_safety": N}}
""").strip()

def llm_judge(transcript, soap_text):
    prompt = _JUDGE_PROMPT.format(transcript=transcript[:3000], soap=soap_text[:2000])
    msg    = [
        {"role": "system", "content": "Respond ONLY with a JSON object. No other text."},
        {"role": "user",   "content": prompt},
    ]
    raw    = _llm(msg, max_tokens=200)
    scores = _parse_json(raw, expect="dict")
    dims   = ["completeness", "correctness", "coherence", "assessment_quality", "plan_safety"]
    for k in dims:
        v = scores.get(k)
        if not isinstance(v, (int, float)) or v < 1 or v > 5:
            scores[k] = 3
        else:
            scores[k] = max(1, min(5, int(v)))
    scores["composite"] = round(sum(scores[k] for k in dims) / len(dims), 2)
    return scores

def check_entity_soap_consistency(soap_text, entities):
    soap_lower = soap_text.lower()
    issues     = []
    confirmed_found = 0
    confirmed_total = 0
    s_section = ""
    s_match = re.search(r'S:\s*(.+?)(?=\n[OAP]:|$)', soap_text, re.DOTALL)
    if s_match:
        s_section = s_match.group(1).lower()
    for e in entities:
        text_lower = e["text"].lower()
        words      = [w for w in text_lower.split() if len(w) >= 4]
        in_soap    = text_lower in soap_lower or any(w in soap_lower for w in words)
        norm       = e.get("normalized", "")
        if norm:
            in_soap = in_soap or norm.lower() in soap_lower
        if e["status"] == "Confirmed" and e["label"] in ("Symptom", "Disease"):
            confirmed_total += 1
            if in_soap:
                confirmed_found += 1
            else:
                issues.append(f"Confirmed {e['label']} '{e['text']}' not found in SOAP")
        if e["status"] == "Negated" and e["label"] == "Symptom":
            if words:
                for w in words:
                    if w in s_section and f"deni" not in s_section[max(0,s_section.find(w)-30):s_section.find(w)]:
                        if w not in {"pain", "chest", "breath", "blood"}:
                            issues.append(f"Negated symptom '{e['text']}' may appear as confirmed in S section")
                            break
    score = confirmed_found / max(confirmed_total, 1)
    return {"score": round(score, 4), "found": confirmed_found, "total": confirmed_total, "issues": issues}

def _split_soap_sections(text):
    if not text:
        return {"S": "", "O": "", "A": "", "P": ""}
    cleaned = re.sub(r'\*\*', '', text)
    cleaned = re.sub(r'#{1,4}\s*', '', cleaned)
    cleaned = re.sub(r'\b(\d\.\s+)?(Subjective)\b',  'S:', cleaned, flags=re.I)
    cleaned = re.sub(r'\b(\d\.\s+)?(Objective)\b',   'O:', cleaned, flags=re.I)
    cleaned = re.sub(r'\b(\d\.\s+)?(Assessment)\b',  'A:', cleaned, flags=re.I)
    cleaned = re.sub(r'\b(\d\.\s+)?(Plan)\b',        'P:', cleaned, flags=re.I)
    sections = {"S": "", "O": "", "A": "", "P": ""}
    pattern  = re.compile(r'(?:^|\n)\s*([SOAP])\s*:\s*', re.MULTILINE)
    matches  = list(pattern.finditer(cleaned))
    for i, m in enumerate(matches):
        key   = m.group(1)
        start = m.end()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned)
        content = cleaned[start:end]
        cut = content.find("[Disclaimer]")
        if cut != -1:
            content = content[:cut]
        if key in sections and not sections[key]:
            sections[key] = content.strip()
    return sections

def evaluate_soap_per_section(candidate_text, reference_text):
    cand_secs = _split_soap_sections(candidate_text)
    ref_secs  = _split_soap_sections(reference_text)
    per_section = {}
    rouge_l_values, rouge_1_values, bleu_values, bert_values = [], [], [], []
    for key in "SOAP":
        c = cand_secs[key]
        r = ref_secs[key]
        if not c.strip() or not r.strip():
            per_section[key] = {
                "rouge1_f1": 0.0, "rouge2_f1": 0.0, "rougeL_f1": 0.0,
                "bleu": 0.0, "bertscore_f1": 0.0,
            }
            continue
        r1 = rouge_n(c, r, n=1)
        r2 = rouge_n(c, r, n=2)
        rl = rouge_l(c, r)
        bl = bleu_score(c, r)
        bs = bertscore_sentence(c, r)
        per_section[key] = {
            "rouge1_f1": r1["f1"], "rouge2_f1": r2["f1"], "rougeL_f1": rl["f1"],
            "bleu": bl["bleu"], "bertscore_f1": bs["f1"],
        }
        rouge_1_values.append(r1["f1"])
        rouge_l_values.append(rl["f1"])
        bleu_values.append(bl["bleu"])
        bert_values.append(bs["f1"])

    def _avg(xs):
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    macro = {
        "rouge1_f1": _avg(rouge_1_values),
        "rougeL_f1": _avg(rouge_l_values),
        "bleu":      _avg(bleu_values),
        "bertscore_f1": _avg(bert_values),
        "sections_compared": len(rouge_l_values),
    }
    return {"per_section": per_section, "macro": macro}


class EvaluationAgent:
    """Computes all evaluation metrics on a SOAP note."""

    def run(self, soap_text: str, transcript: str, entities: list,
            reference_soap: str = None) -> dict:
        import time
        print("  [EvaluationAgent] Running evaluation metrics...")
        t0 = time.time()
        results = {}

        results["keyword_coverage"] = keyword_coverage(soap_text, entities)
        results["hallucination"]    = detect_hallucinations(soap_text, transcript, entities)
        print(f"         Keyword coverage: {results['keyword_coverage']['coverage']:.1%}")
        print(f"         Hallucinations: {results['hallucination']['count']}")

        if reference_soap:
            results["rouge1"]         = rouge_n(soap_text, reference_soap, n=1)
            results["rouge2"]         = rouge_n(soap_text, reference_soap, n=2)
            results["rougeL"]         = rouge_l(soap_text, reference_soap)
            results["bertscore_doc"]  = bertscore_document(soap_text, reference_soap)
            results["bertscore_sent"] = bertscore_sentence(soap_text, reference_soap)
            results["bleu"]           = bleu_score(soap_text, reference_soap)
            r1r = results["rouge1"]["recall"]
            bsr = results["bertscore_sent"]["recall"]
            results["mist_composite"] = round((r1r + bsr) / 2, 4)
            print(f"         ROUGE-1 F1: {results['rouge1']['f1']:.3f}")
            print(f"         BLEU: {results['bleu']['bleu']:.3f} (B1={results['bleu']['bleu1']:.3f} B4={results['bleu']['bleu4']:.3f})")
            print(f"         BERTScore(sent) F1: {results['bertscore_sent']['f1']:.3f}")
            print(f"         MIST Composite: {results['mist_composite']:.3f}")

        print("         Running LLM-as-Judge...")
        results["llm_judge"] = llm_judge(transcript, soap_text)
        print(f"         LLM Judge composite: {results['llm_judge']['composite']}/5.0")

        results["consistency"]  = check_entity_soap_consistency(soap_text, entities)
        print(f"         Entity-SOAP consistency: {results['consistency']['score']:.1%}")

        results["eval_time_s"] = round(time.time() - t0, 2)
        print(f"  [EvaluationAgent] Done in {results['eval_time_s']}s")
        return results
