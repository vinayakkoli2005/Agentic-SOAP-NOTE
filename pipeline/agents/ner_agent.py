import time
import textwrap
from pipeline.utils import (
    _llm, _parse_json, _enforce_schema, _spell_verify_and_align,
    _triple_negation, _is_non_clinical, verify_entities_llm
)
from pipeline.config import USE_NER_VERIFICATION

# ── Exact prompts from notebook Cell 7 ───────────────────────────────────────

_NER_PROMPT = textwrap.dedent("""
You are a clinical Named Entity Recognition engine. Extract ALL clinically relevant
medical entities from the transcript below.

LABELS: Drug | Disease | Symptom | Procedure | NULL
STATUSES: Confirmed | Negated | Historical | Family_History

THINK STEP-BY-STEP INTERNALLY before producing the JSON.
Do NOT include any thinking in the output - return ONLY a JSON array.

RULES:
1. "text" MUST be an EXACT SUBSTRING copied verbatim from the transcript.
2. Extract PERTINENT NEGATIVES: when patient denies a symptom
   (e.g., "Any fevers? No."), extract it with status "Negated".
3. QUALIFIED AFFIRMATIVES are CONFIRMED, not Negated. If the patient gives
   a reason or qualifier ("just from breathing issues", "only when lying down",
   "a little bit"), that is a CONFIRMED symptom. "Just from X" means YES.
4. Substances (tobacco, cigarettes, cannabis, marijuana, alcohol, cocaine,
   opioids) are label "Drug", NOT "Procedure". Smoking is a Drug/substance.
5. Pain MODIFIERS are not entities. Do NOT extract aggravating factors like
   "makes the pain worse", "laying down", "taking a deep breath" as separate
   entities. Only extract the symptom itself (e.g., "chest pain").
6. "Procedure" is ONLY for medical tests, imaging, or referrals the DOCTOR
   explicitly orders (e.g., "EKG", "chest X-ray", "blood work").
7. Extract medications WITH dosage as a single entity when they appear
   together (e.g., "Lisinopril 20 milligrams daily" as one entity).
8. Extract allergies as Disease with status Confirmed.
9. For family history conditions, use status Family_History.
10. Do NOT extract conversational filler, time phrases, or dosage fragments
    without a drug name. Bad: "six weeks ago", "twice a day", "20 milligrams".
11. Extract vital signs and physical exam findings as Symptom (Confirmed).
12. If an entity does not clearly fit Drug, Disease, Symptom, or Procedure,
    label it "NULL". Examples: occupation, living situation, diet, exercise,
    immunization status. Do NOT force a wrong label. NULL beats a wrong label.
13. BE THOROUGH. A typical full encounter has 18-30 entities. Before stopping, scan again for:
    - All confirmed AND denied symptoms (don't skip patient denials)
    - Every medication mentioned (current, historical, denied, OTC)
    - Every disease/condition (current, historical, family history, allergies)
    - Every procedure/test/imaging/referral the doctor mentions
    - Vital signs, exam findings
    - Social history (occupation, smoking, alcohol, drugs)
    - Family history specifics (mother, father, sibling diagnoses)
    If you extract fewer than 15 entities for a normal-length encounter, you have UNDER-EXTRACTED. Re-scan and add the missed ones.
14. "rule out X" / "low suspicion for X" -> status Negated (not Confirmed).
    "concerning for X" / "suggestive of X" -> status Confirmed (positive uncertainty).
15. "used to have X" / "X resolved" / "X cleared up" -> status Historical.
16. ASPIRATIONAL or FUTURE-INTENT mentions are NOT current clinical states. Watch for verbs of desire / planning / consideration ("want to", "hope to", "planning to", "considering", "trying to", "thinking about", "would like to"). When these verbs introduce a clinical outcome, the outcome is the patient's GOAL, not a current Disease / Symptom / Procedure. Either skip the mention entirely, or extract it as NULL Confirmed (a desire / plan), never as the achieved state. Rule of thumb: ask "is the patient describing something that HAS happened / IS happening, or something they WANT to happen?" Only the first qualifies for Disease / Symptom / Procedure.

BAD vs GOOD examples (do NOT do BAD):
BAD : {"text": "weight loss", "label": "Symptom", "status": "Confirmed"} when patient says "I would like to lose weight" - this is a goal, not an actual symptom
GOOD: skip it OR {"text": "lose weight", "label": "NULL", "status": "Confirmed"} (a goal, not a finding)
BAD : {"text": "surgery", "label": "Procedure", "status": "Confirmed"} when patient says "I am considering surgery" - the procedure has NOT been ordered or performed
GOOD: skip it (no procedure has actually been ordered)
BAD : {"text": "twice a day", "label": "Drug"} - dosage fragment, no drug
GOOD: skip it
BAD : {"text": "shoveling", "label": "Symptom"} - activity, not symptom
GOOD: skip it
BAD : {"text": "smoke", "label": "Procedure"} - substance use is Drug
GOOD: {"text": "smoke", "label": "Drug", "status": "Confirmed"}
BAD : {"text": "heart attack", "label": "Disease", "status": "Confirmed"} - when father had it
GOOD: {"text": "heart attack", "label": "Disease", "status": "Family_History"}

EXAMPLE 1 - basic negatives + affirmatives:
Transcript: "Do you have chest pain? Yes. Any fevers? No. Sweaty? Just from breathing."
Output:
[
  {"text": "chest pain", "label": "Symptom", "status": "Confirmed"},
  {"text": "fevers", "label": "Symptom", "status": "Negated"},
  {"text": "Sweaty", "label": "Symptom", "status": "Confirmed"}
]

EXAMPLE 2 - drugs, family, social:
Transcript: "I take Metformin 500mg. My father had a heart attack. I smoke a pack a day. I'm an accountant."
Output:
[
  {"text": "Metformin 500mg", "label": "Drug", "status": "Confirmed"},
  {"text": "heart attack", "label": "Disease", "status": "Family_History"},
  {"text": "smoke", "label": "Drug", "status": "Confirmed"},
  {"text": "accountant", "label": "NULL", "status": "Confirmed"}
]

EXAMPLE 3 - negation cluster + historical:
Transcript: "Patient denies fever. We'll work up for pulmonary embolism. She had pneumonia two years ago that resolved."
Output:
[
  {"text": "fever", "label": "Symptom", "status": "Negated"},
  {"text": "pulmonary embolism", "label": "Disease", "status": "Negated"},
  {"text": "pneumonia", "label": "Disease", "status": "Historical"}
]

OUTPUT: Return ONLY a JSON array. No other text.
""").strip()

_NER_RETRY_PROMPT = textwrap.dedent("""
Extract medical entities from the transcript as a JSON array.
Each entity: {"text": "exact words from transcript", "label": "Drug|Disease|Symptom|Procedure|NULL", "status": "Confirmed|Negated|Historical|Family_History"}
Include symptoms the patient confirms AND denies.
Return ONLY the JSON array.
""").strip()


class NERAgent:
    """Extracts clinical entities using SPELL architecture with verification passes."""

    def run(self, transcript: str):
        print("  [NERAgent] Phase 1: LLM extraction (3-shot)...")
        t0 = time.time()

        msg = [
            {"role": "system", "content": _NER_PROMPT},
            {"role": "user", "content": (
                f"Transcript:\n{transcript}\n\n"
                "Extract ALL entities. EXACT SUBSTRINGS only. JSON array:"
            )}
        ]
        raw = _llm(msg, max_tokens=3500, temperature=0.1)
        candidates = _parse_json(raw, "list")
        print(f"         -> {len(candidates)} raw candidates in {time.time()-t0:.1f}s")

        if len(candidates) == 0:
            print("  [NERAgent] RETRY: 0 candidates, using simpler prompt...")
            retry_msg = [
                {"role": "system", "content": _NER_RETRY_PROMPT},
                {"role": "user", "content": f"Transcript:\n{transcript}\n\nJSON array:"}
            ]
            raw = _llm(retry_msg, max_tokens=3000, temperature=0.1)
            candidates = _parse_json(raw, "list")
            print(f"         -> {len(candidates)} candidates after retry in {time.time()-t0:.1f}s")

        schema_clean, schema_rejected = _enforce_schema(candidates)

        print("  [NERAgent] Phase 3: Non-clinical filter (regex only)...")
        clinical, nonclinical_rejected = [], []
        for ent in schema_clean:
            if _is_non_clinical(ent["text"]):
                nonclinical_rejected.append({**ent, "_reason": "Non-clinical (regex)"})
                print(f"         JUNK: \"{ent['text']}\"")
                continue
            clinical.append(ent)

        print("  [NERAgent] Phase 4: SPELL verification...")
        verified, spell_rejected = [], []
        for ent in clinical:
            offset = _spell_verify_and_align(ent["text"], transcript)
            if offset is not None:
                ent["first_offset"] = offset
                verified.append(ent)
            else:
                spell_rejected.append({**ent, "_reason": "SPELL: not in transcript"})
                print(f"         REJECTED: \"{ent['text']}\"")

        print("  [NERAgent] Phase 5: Offset-aware negation (hardened)...")
        for e in verified:
            old = e["status"]
            e["status"] = _triple_negation(
                e["text"], transcript, old, e["label"],
                spell_offset=e.get("first_offset")
            )
            if e["status"] != old:
                e["_neg_fix"] = f"{old}->{e['status']}"

        verify_rejected = []
        if USE_NER_VERIFICATION:
            print("  [NERAgent] Phase 6: LLM verification (VerifyNER-style)...")
            verified, verify_rejected = verify_entities_llm(verified, transcript, batch_size=15)

        all_rejected = schema_rejected + nonclinical_rejected + spell_rejected + verify_rejected
        print(f"  [NERAgent] Final: {len(verified)} verified, {len(all_rejected)} rejected")
        return verified, all_rejected
