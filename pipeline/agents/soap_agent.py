import re
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

import pipeline.models as models
from pipeline.utils import _llm, _parse_json, sanitize_assessment
from pipeline.config import USE_SELF_CONSISTENCY, SELF_CONSISTENCY_N

# ── Fact table builder (exact from notebook Cell 8) ───────────────────────────

def build_fact_table(entities, transcript):
    facts = {
        "confirmed_symptoms": [], "negated_symptoms": [], "diseases": [],
        "family_history": [], "historical": [], "medications_current": [],
        "medications_denied": [], "allergies": [], "social_history": [],
        "vitals_and_exam": [], "procedures_ordered": [], "procedures_denied": [],
        "unclassified": [],
    }
    for e in entities:
        if e.get("label") == "NULL":
            text   = e.get("text", "").strip()
            status = e.get("status", "Confirmed")
            if status == "Confirmed":
                facts["unclassified"].append(text)

    social_keywords = {"smoke","cigarette","tobacco","cannabis","marijuana",
                       "alcohol","drink","cocaine","meth","opioid","heroin",
                       "iv drug","recreational"}

    for e in entities:
        if e.get("label") == "NULL":
            continue
        text   = e["text"]
        label  = e["label"]
        status = e["status"]
        norm   = e.get("normalized")
        display = norm if (norm and norm.lower() != text.lower()) else text

        offset  = e.get("first_offset")
        context = ""
        if offset and offset.get("start") is not None:
            s   = max(0, offset["start"] - 60)
            end = min(len(transcript), offset["end"] + 60)
            context = transcript[s:end].strip()

        entry = display
        if context and status == "Confirmed" and label == "Symptom":
            entry = f"{display} [context: {context}]"

        if label == "Symptom":
            if status == "Confirmed":     facts["confirmed_symptoms"].append(entry)
            elif status == "Negated":     facts["negated_symptoms"].append(display)
            elif status == "Family_History": facts["family_history"].append(display)
            elif status == "Historical":  facts["historical"].append(display)

        elif label == "Disease":
            if status == "Confirmed":     facts["diseases"].append(display)
            elif status == "Family_History": facts["family_history"].append(display)
            elif status == "Historical":  facts["historical"].append(display)

        elif label == "Drug":
            is_social = any(kw in text.lower() for kw in social_keywords)
            if is_social:
                if status == "Confirmed":
                    facts["social_history"].append(f"{text} [context: {context}]" if context else text)
                elif status == "Negated":
                    facts["social_history"].append(f"Denies {text}")
            else:
                if status == "Confirmed":  facts["medications_current"].append(text)
                elif status == "Negated":  facts["medications_denied"].append(text)

        elif label == "Procedure":
            if status == "Confirmed":    facts["procedures_ordered"].append(text)
            elif status == "Negated":    facts["procedures_denied"].append(text)

    lines = []
    for key, items in facts.items():
        header = key.replace("_", " ").upper()
        if items:
            lines.append(f"{header}:")
            for item in items:
                lines.append(f"  - {item}")
        else:
            lines.append(f"{header}: None documented")
    return "\n".join(lines), facts


# ── SOAP body prompts (exact from notebook Cell 8) ────────────────────────────

_SOAP_BODY_SYSTEM = """You are a strict EXTRACTIVE Clinical Scribe. You PARAPHRASE the structured fact table into SOAP format. You NEVER add medical reasoning, diagnoses, or speculation.

OUTPUT FORMAT (plain text, no markdown, no bold, no bullet symbols):
S: Subjective - paraphrase the confirmed symptoms, history, and social context from the fact table. End with pertinent negatives ("Denies X, Y, Z") from the negated_symptoms section.
O: Objective - paraphrase ONLY items from the vitals_and_exam section. If none, write exactly: No objective findings recorded in transcript.
A: Assessment - a NEUTRAL SUMMARY of the patient's confirmed clinical picture. Do NOT diagnose. Do NOT use words like "suspected", "likely", "rule out", "concerning for", "consistent with". Just summarize: "Patient presents with [confirmed symptoms]. Relevant history includes [historical/family]. Current medications: [meds]."
P: Plan - paraphrase ONLY items from the procedures_ordered section. If none, write exactly: No specific medical orders were recorded in the transcript.

THINK STEP-BY-STEP INTERNALLY before writing. Do NOT include the thinking - output only the four SOAP lines.

ABSOLUTE RULES:
1. EVERY claim in S, O, A, P must come from the fact table. If it is not in the table, it does NOT appear.
2. Denied symptoms go in S as pertinent negatives only ("Denies fever, vomiting, chest pain").
3. The Assessment is descriptive ONLY, never diagnostic.
4. No "[Disclaimer]" line, no "AI Suggestion" - those are generated separately.
5. Plain text. No **bold**. No #headings. No bullet points (-, *, .).

BAD vs GOOD examples (NEVER do BAD):
BAD : A: Concerning for acute coronary syndrome. Recommend rule out MI.
GOOD: A: Patient presents with exertional chest pressure, dyspnea, and family history of premature MI. Currently taking Lisinopril 10mg and Aspirin 81mg.
BAD : P: Standard workup should include serial troponins, lipid panel, stress testing.
GOOD: P: Doctor ordered EKG, serum troponin levels, and chest X-ray.
BAD : O: Vital signs appear stable, patient is in no acute distress.
GOOD: O: Blood pressure 148/92 mmHg. Heart rate 88 bpm."""

_SOAP_BODY_USER = """FACT TABLE:
{fact_table}

Write the SOAP note now. S, O, A, P only. No markdown, no disclaimers, no AI suggestions.
S:"""

_AI_SUGGESTION_SYSTEM = """You are a clinical AI providing brief decision support to a physician AFTER they have written their SOAP note. This is the ONLY place clinical reasoning belongs.

OUTPUT FORMAT (plain text, 4 short lines, no markdown, no bullets, total under 80 words):
Differential: <1-2 most likely diagnoses, comma-separated, no reasoning>
Workup: <1-3 specific tests or actions, comma-separated>
Red flags: <symptoms warranting urgent escalation, or "None">
Follow-up: <appropriate timeframe>

HARD RULES:
- Be BRIEF and SPECIFIC to the SOAP content. No generic boilerplate.
- If the SOAP note is sparse or empty, output exactly: "Insufficient SOAP detail for clinical suggestions."
- Plain text only. No bold, no markdown headers."""

_AI_SUGGESTION_USER = """Here is the SOAP note for the encounter:

{soap_body}

Now generate the AI Suggestion section."""


def _parse_soap_sections(raw_text):
    cleaned = re.sub(r'\*\*', '', raw_text)
    cleaned = re.sub(r'#{1,4}\s*', '', cleaned)
    soap    = {"S": "", "O": "", "A": "", "P": ""}
    pattern = re.compile(r'(?:^|\n)\s*([SOAP])\s*:\s*', re.MULTILINE)
    matches = list(pattern.finditer(cleaned))
    for i, m in enumerate(matches):
        key   = m.group(1)
        start = m.end()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned)
        content = cleaned[start:end].strip()
        if key in soap and not soap[key]:
            soap[key] = content
    return soap


def _score_soap_candidate(soap, facts):
    soap_text = " ".join(soap.get(k, "") for k in "SOAP").lower()
    hallucination_phrases = [
        "suspected","likely","concerning for","rule out","consistent with",
        "in keeping with","acute coronary syndrome","myocardial infarction",
        "suggestive of","differential diagnosis","workup includes",
    ]
    hallucination_count = sum(1 for p in hallucination_phrases if p in soap_text)
    all_facts = []
    for k in ("confirmed_symptoms","diseases","medications_current","vitals_and_exam","procedures_ordered"):
        all_facts.extend(facts.get(k, []))
    if not all_facts:
        coverage = 1.0
    else:
        found = 0
        for fact in all_facts:
            key = re.sub(r"\[context:.*?\]", "", str(fact)).strip().lower()
            words = [w for w in key.split() if len(w) >= 4]
            if not words:
                continue
            if any(w in soap_text for w in words[:3]):
                found += 1
        coverage = found / max(len(all_facts), 1)
    score = hallucination_count - coverage
    return score, hallucination_count, coverage


def _hard_default_sections(soap, facts):
    if not facts.get("vitals_and_exam"):
        soap["O"] = "No objective findings recorded in transcript."
    if not facts.get("procedures_ordered"):
        soap["P"] = "No specific medical orders were recorded in the transcript."
    if not soap.get("S"):
        s_parts = []
        if facts.get("confirmed_symptoms"):
            s_parts.append("Patient reports " + ", ".join(
                re.sub(r"\s*\[context:.*?\]", "", x).strip()
                for x in facts["confirmed_symptoms"]) + ".")
        if facts.get("historical"):
            s_parts.append("History of " + ", ".join(facts["historical"]) + ".")
        if facts.get("family_history"):
            s_parts.append("Family history of " + ", ".join(facts["family_history"]) + ".")
        if facts.get("social_history"):
            s_parts.append("Social history: " + "; ".join(facts["social_history"]) + ".")
        if facts.get("medications_current"):
            s_parts.append("Current medications: " + ", ".join(facts["medications_current"]) + ".")
        if facts.get("negated_symptoms"):
            s_parts.append("Denies " + ", ".join(facts["negated_symptoms"]) + ".")
        soap["S"] = " ".join(s_parts) if s_parts else "No subjective findings recorded in transcript."
    if not soap.get("A"):
        a_parts = []
        if facts.get("confirmed_symptoms"):
            a_parts.append("Patient presents with " + ", ".join(
                re.sub(r"\s*\[context:.*?\]", "", x).strip()
                for x in facts["confirmed_symptoms"]) + ".")
        if facts.get("diseases"):
            a_parts.append("Reported diagnoses: " + ", ".join(facts["diseases"]) + ".")
        if facts.get("family_history"):
            a_parts.append("Notable family history: " + ", ".join(facts["family_history"]) + ".")
        if facts.get("medications_current"):
            a_parts.append("Current medications: " + ", ".join(facts["medications_current"]) + ".")
        soap["A"] = " ".join(a_parts) if a_parts else "No assessment-relevant findings recorded in transcript."
    if not soap.get("P") and facts.get("procedures_ordered"):
        soap["P"] = "Doctor ordered " + ", ".join(facts["procedures_ordered"]) + "."
    return soap


def _generate_soap_body_once(fact_table, facts, seed=None):
    messages = [
        {"role": "system", "content": _SOAP_BODY_SYSTEM},
        {"role": "user",   "content": _SOAP_BODY_USER.format(fact_table=fact_table)},
    ]
    raw  = _llm(messages, max_tokens=900, temperature=0.1, seed=seed)
    soap = _parse_soap_sections(raw)
    soap = _hard_default_sections(soap, facts)
    score, halluc, cov = _score_soap_candidate(soap, facts)
    return soap, raw, score, halluc, cov


def _generate_ai_suggestion(soap_body_text):
    placeholders = (
        "No subjective findings recorded in transcript.",
        "No objective findings recorded in transcript.",
        "No specific medical orders were recorded in the transcript.",
    )
    stripped = soap_body_text
    for p in placeholders:
        stripped = stripped.replace(p, "")
    stripped = re.sub(r'[KSOAP]:\s*', '', stripped).strip()
    if len(stripped) < 40:
        return "Insufficient SOAP detail for clinical suggestions."
    msg = [
        {"role": "system", "content": _AI_SUGGESTION_SYSTEM},
        {"role": "user",   "content": _AI_SUGGESTION_USER.format(soap_body=soap_body_text)},
    ]
    raw     = _llm(msg, max_tokens=220, temperature=0.2)
    cleaned = re.sub(r'\*\*', '', raw).strip()
    cleaned = re.sub(r'^#{1,4}\s*', '', cleaned, flags=re.MULTILINE)
    return cleaned


# ── Few-shot bank (exact from notebook Cell 13) ───────────────────────────────

FEWSHOT_BANK = [
    {
        "id": "cardiac_01", "specialty": "Cardiology",
        "conversation": (
            "Doctor: What brings you in today? Patient: I've been having chest pain "
            "for the past three days. It feels like a pressure on my chest, especially "
            "when I walk or climb stairs. Doctor: Does it go away when you rest? "
            "Patient: Yes, usually within a few minutes. Doctor: Any shortness of "
            "breath? Patient: Yes, I get winded easily now. Doctor: Any history of "
            "heart problems? Patient: My father had a heart attack at 55. Doctor: Do "
            "you smoke? Patient: I quit two years ago but I smoked for 20 years. "
            "Doctor: Any medications? Patient: I take Lisinopril 10 milligrams daily "
            "and Aspirin 81 milligrams. Doctor: Let me check your vitals. Blood "
            "pressure is 148 over 92, heart rate 88. I'm going to order an EKG and "
            "troponin levels. I'd also like to get a chest X-ray."
        ),
        "ksoap": (
            'K: ("chest pain", Chest pain) [Symptom, Confirmed], '
            '("shortness of breath", Dyspnea) [Symptom, Confirmed], '
            '("heart attack", Myocardial infarction) [Disease, Family_History], '
            '("smoked for 20 years", Tobacco use) [Social, Historical], '
            '("Lisinopril 10 milligrams daily", Lisinopril) [Drug, Confirmed], '
            '("Aspirin 81 milligrams", Aspirin) [Drug, Confirmed], '
            '("148 over 92", Elevated blood pressure) [Vital, Confirmed], '
            '("heart rate 88", Heart rate) [Vital, Confirmed], '
            '("EKG", Electrocardiogram) [Procedure, Confirmed], '
            '("troponin levels", Troponin measurement) [Procedure, Confirmed], '
            '("chest X-ray", Chest radiograph) [Procedure, Confirmed]\n\n'
            "S: Patient presents with a three-day history of exertional chest "
            "pressure that resolves with rest. Reports associated dyspnea on "
            "exertion. Significant family history of premature coronary artery "
            "disease (father with MI at age 55). Former smoker with 20-pack-year "
            "history, quit two years ago. Currently taking Lisinopril 10mg daily "
            "and Aspirin 81mg daily.\n\n"
            "O: Blood pressure 148/92 mmHg (elevated). Heart rate 88 bpm.\n\n"
            "A: Exertional chest pain consistent with stable angina pectoris. "
            "Significant cardiac risk factors include hypertension, extensive "
            "smoking history, and family history of premature MI. Differential "
            "diagnoses include unstable angina and musculoskeletal chest wall pain.\n\n"
            "P: Doctor ordered EKG, serum troponin levels, and chest X-ray. "
            "[Disclaimer] AI Suggestion: Standard workup includes serial troponins, "
            "lipid panel, stress testing, and cardiology referral if troponins positive."
        ),
    },
    {
        "id": "gi_01", "specialty": "Gastroenterology",
        "conversation": (
            "Doctor: What's going on today? Patient: I've had really bad diarrhea "
            "for about five days now. Doctor: How many times a day? Patient: Like "
            "six or seven times. It's watery. Doctor: Any blood in the stool? "
            "Patient: No, no blood. Doctor: Any fever? Patient: Yeah, I had a "
            "fever of 101 two days ago. Doctor: Nausea or vomiting? Patient: Some "
            "nausea but no vomiting. Doctor: Any belly pain? Patient: Yeah, crampy "
            "pain around my belly button. Doctor: Any medications? Patient: No. "
            "Doctor: Allergies? Patient: None. Doctor: Abdomen is soft, diffusely "
            "tender, hyperactive bowel sounds. I'm going to send a stool culture "
            "and order a basic metabolic panel."
        ),
        "ksoap": (
            'K: ("diarrhea", Diarrhea) [Symptom, Confirmed], '
            '("blood in the stool", Rectal bleeding) [Symptom, Negated], '
            '("fever of 101", Fever) [Symptom, Confirmed], '
            '("nausea", Nausea) [Symptom, Confirmed], '
            '("no vomiting", Vomiting) [Symptom, Negated], '
            '("crampy pain around my belly button", Abdominal cramps) [Symptom, Confirmed], '
            '("diffusely tender", Abdominal tenderness) [Symptom, Confirmed], '
            '("hyperactive bowel sounds", Hyperactive bowel sounds) [Symptom, Confirmed], '
            '("stool culture", Stool culture) [Procedure, Confirmed], '
            '("basic metabolic panel", BMP) [Procedure, Confirmed]\n\n'
            "S: Patient reports five days of watery diarrhea six to seven times "
            "daily without hematochezia. Associated fever (101F two days ago), "
            "nausea without emesis, and periumbilical crampy abdominal pain. "
            "Denies medications and allergies.\n\n"
            "O: Abdomen soft, diffusely tender. Hyperactive bowel sounds.\n\n"
            "A: Acute infectious gastroenteritis with dehydration risk given "
            "frequency of diarrhea and fever. Differential includes viral "
            "gastroenteritis and bacterial food poisoning.\n\n"
            "P: Doctor ordered stool culture and basic metabolic panel. "
            "[Disclaimer] AI Suggestion: Oral rehydration therapy, BRAT diet, "
            "empiric antibiotics if culture positive."
        ),
    },
    {
        "id": "resp_01", "specialty": "Pulmonology",
        "conversation": (
            "Doctor: Tell me what's been happening. Patient: I've had this terrible "
            "cough for two weeks. It started dry but now I'm coughing up green mucus. "
            "Doctor: Any fever? Patient: Yes, about 100.8. Doctor: Shortness of "
            "breath? Patient: Yes, especially lying down at night. Doctor: Any "
            "wheezing? Patient: Yes. Doctor: Do you have asthma? Patient: I had "
            "mild asthma as a kid but haven't needed an inhaler in years. Doctor: "
            "Do you smoke? Patient: No, never. Doctor: I'm hearing crackles in your "
            "right lower lobe. Oxygen saturation is 94 percent. I'm going to order "
            "a chest X-ray and start you on antibiotics."
        ),
        "ksoap": (
            'K: ("cough", Cough) [Symptom, Confirmed], '
            '("green mucus", Productive cough) [Symptom, Confirmed], '
            '("100.8", Fever) [Symptom, Confirmed], '
            '("shortness of breath", Dyspnea) [Symptom, Confirmed], '
            '("wheezing", Wheezing) [Symptom, Confirmed], '
            '("asthma as a kid", Asthma) [Disease, Historical], '
            '("No, never", Tobacco use) [Social, Negated], '
            '("crackles in your right lower lobe", Pulmonary crackles) [Symptom, Confirmed], '
            '("94 percent", Oxygen saturation) [Vital, Confirmed], '
            '("chest X-ray", Chest radiograph) [Procedure, Confirmed], '
            '("antibiotics", Antibiotic therapy) [Drug, Confirmed]\n\n'
            "S: Two-week cough progressing from dry to productive with green sputum. "
            "Fever 100.8F, dyspnea worse supine, audible wheezing. Childhood asthma "
            "history, currently not on inhalers. Non-smoker.\n\n"
            "O: Crackles in right lower lobe on auscultation. Oxygen saturation "
            "94% on room air.\n\n"
            "A: Right lower lobe community-acquired pneumonia as evidenced by "
            "productive cough with purulent sputum, focal crackles, fever, and "
            "mild hypoxemia. Differential includes acute bronchitis and asthma "
            "exacerbation with superimposed infection.\n\n"
            "P: Doctor ordered chest X-ray and initiated antibiotic therapy. "
            "[Disclaimer] AI Suggestion: Amoxicillin-clavulanate or azithromycin. "
            "Consider bronchodilator given asthma history."
        ),
    },
    {
        "id": "obgyn_01", "specialty": "OB/GYN",
        "conversation": (
            "Doctor: What brings you in? Patient: I've been feeling really nauseous "
            "and I missed my period. Doctor: When was your last period? Patient: "
            "About seven weeks ago. Doctor: Are your periods usually regular? "
            "Patient: Yes, every 28 days usually. Doctor: Any vomiting? Patient: "
            "Yes, mostly in the mornings. Doctor: Breast tenderness? Patient: "
            "Actually yes. Doctor: Are you sexually active? Patient: Yes. Doctor: "
            "What birth control do you use? Patient: We use condoms but not every "
            "time. Doctor: Let me order a urine pregnancy test."
        ),
        "ksoap": (
            'K: ("nauseous", Nausea) [Symptom, Confirmed], '
            '("missed my period", Amenorrhea) [Symptom, Confirmed], '
            '("vomiting", Vomiting) [Symptom, Confirmed], '
            '("Breast tenderness", Breast tenderness) [Symptom, Confirmed], '
            '("condoms", Condom use) [Drug, Confirmed], '
            '("urine pregnancy test", Pregnancy test) [Procedure, Confirmed]\n\n'
            "S: Patient reports nausea with morning vomiting and missed menstrual "
            "period. LMP approximately seven weeks ago with normally regular 28-day "
            "cycles. Reports breast tenderness. Sexually active with inconsistent "
            "condom use.\n\n"
            "O: No physical exam findings or vitals recorded.\n\n"
            "A: Suspected early pregnancy based on amenorrhea, morning emesis, and "
            "breast tenderness in reproductive-age female with inconsistent "
            "contraception. Differential includes ectopic pregnancy.\n\n"
            "P: Doctor ordered urine pregnancy test. [Disclaimer] AI Suggestion: "
            "If positive, quantitative beta-hCG, prenatal labs, dating ultrasound, "
            "and prenatal vitamins with folic acid."
        ),
    },
    {
        "id": "msk_01", "specialty": "Orthopedics",
        "conversation": (
            "Doctor: What happened? Patient: I twisted my ankle playing basketball "
            "yesterday. It swelled up right away. Doctor: Can you put weight on it? "
            "Patient: I can walk but it hurts a lot. Doctor: Any numbness or "
            "tingling? Patient: No. Doctor: Have you injured this ankle before? "
            "Patient: No. Doctor: There's swelling and tenderness over the lateral "
            "malleolus. No deformity. Pedal pulses intact. I'm ordering an X-ray "
            "of your left ankle."
        ),
        "ksoap": (
            'K: ("twisted my ankle", Ankle sprain) [Symptom, Confirmed], '
            '("swelled up", Edema) [Symptom, Confirmed], '
            '("numbness", Numbness) [Symptom, Negated], '
            '("tingling", Paresthesia) [Symptom, Negated], '
            '("tenderness over the lateral malleolus", Lateral malleolus tenderness) [Symptom, Confirmed], '
            '("X-ray of your left ankle", Ankle radiograph) [Procedure, Confirmed]\n\n'
            "S: Acute left ankle injury from basketball the prior day with immediate "
            "swelling. Weight-bearing with significant pain. Denies numbness and "
            "tingling. No prior ankle injuries.\n\n"
            "O: Left ankle swelling and tenderness over lateral malleolus. No "
            "deformity. Pedal pulses intact.\n\n"
            "A: Acute left lateral ankle sprain, likely ATFL injury. Ottawa Ankle "
            "Rules positive given lateral malleolus tenderness. Differential "
            "includes lateral malleolus fracture.\n\n"
            "P: Doctor ordered left ankle X-ray. [Disclaimer] AI Suggestion: "
            "RICE protocol, NSAIDs, air-stirrup brace. Orthopedic referral if fracture."
        ),
    },
    {
        "id": "neuro_01", "specialty": "Neurology",
        "conversation": (
            "Doctor: What's troubling you? Patient: I've been getting terrible "
            "headaches for about a month. Doctor: Where is the pain? Patient: "
            "Right side, behind my eye. Doctor: How often? Patient: Three times a "
            "week. Doctor: Any nausea? Patient: Yes, and sometimes flashing lights "
            "before they start. Doctor: How long do they last? Patient: Four to six "
            "hours. Doctor: Family history of migraines? Patient: My mother gets "
            "them. Doctor: Medications? Patient: Ibuprofen but it doesn't help "
            "anymore. Doctor: I'm starting you on Sumatriptan for acute episodes."
        ),
        "ksoap": (
            'K: ("headaches", Headache) [Symptom, Confirmed], '
            '("nausea", Nausea) [Symptom, Confirmed], '
            '("flashing lights", Visual aura) [Symptom, Confirmed], '
            '("My mother gets them", Migraine) [Disease, Family_History], '
            '("Ibuprofen", Ibuprofen) [Drug, Confirmed], '
            '("Sumatriptan", Sumatriptan) [Drug, Confirmed]\n\n'
            "S: One-month history of recurrent right-sided retro-orbital headaches "
            "three times weekly, lasting four to six hours. Associated nausea and "
            "visual aura. Family history of migraines in mother. Ibuprofen no "
            "longer effective.\n\n"
            "O: No physical exam findings or vitals recorded.\n\n"
            "A: Migraine with aura meeting ICHD-3 criteria given recurrent "
            "unilateral headaches with visual aura, nausea, and positive family "
            "history. Differential includes tension-type and cluster headache.\n\n"
            "P: Doctor prescribed Sumatriptan for acute migraine episodes. "
            "[Disclaimer] AI Suggestion: Consider preventive therapy (topiramate, "
            "propranolol). Headache diary for trigger identification."
        ),
    },
    {
        "id": "infectious_01", "specialty": "Internal Medicine",
        "conversation": (
            "Doctor: What's going on? Patient: Sore throat and fever for three "
            "days. Doctor: How high? Patient: About 102. Doctor: Any cough? "
            "Patient: No. Doctor: Body aches? Patient: Yes. Doctor: Rash? Patient: "
            "No. Doctor: Your pharynx is erythematous with tonsillar exudates. "
            "Tender anterior cervical lymphadenopathy. I'm doing a rapid strep test."
        ),
        "ksoap": (
            'K: ("Sore throat", Pharyngitis) [Symptom, Confirmed], '
            '("fever", Fever) [Symptom, Confirmed], '
            '("cough", Cough) [Symptom, Negated], '
            '("Body aches", Myalgia) [Symptom, Confirmed], '
            '("Rash", Rash) [Symptom, Negated], '
            '("erythematous", Pharyngeal erythema) [Symptom, Confirmed], '
            '("tonsillar exudates", Tonsillar exudate) [Symptom, Confirmed], '
            '("cervical lymphadenopathy", Cervical lymphadenopathy) [Symptom, Confirmed], '
            '("rapid strep test", Rapid streptococcal antigen test) [Procedure, Confirmed]\n\n'
            "S: Three-day sore throat and fever to 102F. Diffuse body aches. "
            "Denies cough and rash.\n\n"
            "O: Erythematous pharynx with bilateral tonsillar exudates. Tender "
            "anterior cervical lymphadenopathy.\n\n"
            "A: Acute pharyngitis with high suspicion for Group A Streptococcal "
            "infection. Centor score 4/4 (fever, exudates, lymphadenopathy, no "
            "cough). Differential includes mononucleosis.\n\n"
            "P: Doctor ordered rapid strep test. [Disclaimer] AI Suggestion: "
            "If positive, penicillin V or amoxicillin 10 days. Monospot if negative."
        ),
    },
    {
        "id": "endo_01", "specialty": "Endocrinology",
        "conversation": (
            "Doctor: How are you feeling? Patient: Very thirsty lately and going "
            "to the bathroom all the time. Doctor: How often? Patient: Probably "
            "every hour. Doctor: Weight loss? Patient: About ten pounds in a month "
            "without trying. Doctor: Blurry vision? Patient: A little. Doctor: "
            "Family history of diabetes? Patient: My father has type 2. Doctor: "
            "Your fasting glucose is 285. I need to check your hemoglobin A1c. "
            "I'm starting you on Metformin 500 milligrams twice daily."
        ),
        "ksoap": (
            'K: ("Very thirsty", Polydipsia) [Symptom, Confirmed], '
            '("bathroom all the time", Polyuria) [Symptom, Confirmed], '
            '("ten pounds in a month", Unintentional weight loss) [Symptom, Confirmed], '
            '("Blurry vision", Blurred vision) [Symptom, Confirmed], '
            '("father has type 2", Diabetes mellitus type 2) [Disease, Family_History], '
            '("fasting glucose is 285", Hyperglycemia) [Vital, Confirmed], '
            '("hemoglobin A1c", HbA1c measurement) [Procedure, Confirmed], '
            '("Metformin 500 milligrams twice daily", Metformin) [Drug, Confirmed]\n\n'
            "S: Polydipsia, polyuria (hourly), unintentional 10-pound weight loss "
            "over one month, blurry vision. Family history of type 2 diabetes in "
            "father.\n\n"
            "O: Fasting blood glucose 285 mg/dL (severely elevated, normal <100).\n\n"
            "A: New-onset type 2 diabetes mellitus with classic presentation of "
            "polyuria, polydipsia, weight loss, and fasting glucose 285. Visual "
            "changes may indicate hyperglycemia-related lens changes.\n\n"
            "P: Doctor ordered hemoglobin A1c and started Metformin 500mg twice "
            "daily. [Disclaimer] AI Suggestion: CMP, lipid panel, urine "
            "microalbumin, ophthalmology referral, diabetes education."
        ),
    },
]


def _build_fewshot_embeddings():
    texts = [ex["conversation"] for ex in FEWSHOT_BANK]
    return np.array(models.embedder.encode(texts, convert_to_tensor=False, show_progress_bar=False))

def _select_fewshot_smallest(embeddings, k=3):
    if k >= len(FEWSHOT_BANK):
        return list(range(len(FEWSHOT_BANK)))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    indices = []
    for i in range(k):
        cluster_idx = np.where(labels == i)[0]
        if len(cluster_idx) == 0:
            continue
        lengths   = np.array([len(FEWSHOT_BANK[j]["conversation"]) for j in cluster_idx])
        best_local = int(np.argmin(lengths))
        indices.append(int(cluster_idx[best_local]))
    return indices

def _select_fewshot_query_aware(embeddings, query_text, k=3):
    if k >= len(FEWSHOT_BANK):
        return list(range(len(FEWSHOT_BANK)))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    query_emb = models.embedder.encode([query_text], convert_to_tensor=False)
    indices = []
    for i in range(k):
        cluster_idx = np.where(labels == i)[0]
        if len(cluster_idx) == 0:
            continue
        cluster_embs = embeddings[cluster_idx]
        sims = sklearn_cosine(cluster_embs, query_emb).flatten()
        best_local = int(np.argmax(sims))
        indices.append(int(cluster_idx[best_local]))
    return indices


def build_keyword_section(entities):
    if not entities:
        return "No clinical keywords extracted."
    parts = []
    for e in entities:
        span = e.get("text", "").strip()
        if not span:
            continue
        normalized = e.get("normalized")
        snomed     = e.get("snomed")
        fast_norm  = None
        from pipeline.agents.code_enrichment_agent import _FAST_NORMALIZE
        fast_norm = _FAST_NORMALIZE.get(span.lower())
        if normalized and normalized.lower() != span.lower():
            entity_name = normalized.title()
        elif snomed and snomed.get("term"):
            entity_name = snomed["term"]
        elif fast_norm:
            entity_name = fast_norm.title()
        else:
            entity_name = span.title() if len(span) < 40 else span[:40].title()
        label  = e.get("label", "Symptom")
        status = e.get("status", "Confirmed")
        parts.append(f'("{span}", {entity_name}) [{label}, {status}]')
    return ", ".join(parts) if parts else "No clinical keywords extracted."


def _build_ksoap_body_prompt(examples, fact_table, keyword_section):
    system = (
        "You are a strict EXTRACTIVE Clinical Scribe writing K-SOAP notes.\n"
        "You PARAPHRASE the structured fact table into the K-SOAP format.\n"
        "You NEVER add diagnoses, speculation, or reasoning beyond what is in the facts.\n\n"
        "K-SOAP FORMAT (plain text, no markdown):\n"
        'K: Keywords in ("span", Entity) [Label, Assertion] format - PROVIDED below, copy VERBATIM.\n'
        "S: Subjective - paraphrase confirmed symptoms, history, social context. End with pertinent negatives.\n"
        "O: Objective - paraphrase ONLY vitals/exam from the fact table. If none, write exactly: No objective findings recorded in transcript.\n"
        "A: Assessment - NEUTRAL SUMMARY of confirmed clinical picture. Do NOT diagnose. Do NOT use 'suspected', 'likely', 'rule out', 'concerning for'.\n"
        "P: Plan - paraphrase ONLY procedures_ordered from the fact table. If none, write exactly: No specific medical orders were recorded in the transcript.\n\n"
        "ABSOLUTE RULES:\n"
        "1. Copy the K section verbatim from the keyword list below.\n"
        "2. No markdown. No **bold**. Plain text K:, S:, O:, A:, P: prefixes only.\n"
        "3. EVERY claim in S/O/A/P must come from the fact table.\n"
        "4. Do NOT include any [Disclaimer] or AI Suggestion - those are generated separately.\n"
        "5. STOP after the P: line.\n\n"
        "BAD : A: Likely acute coronary syndrome; consider stress test.\n"
        "GOOD: A: Patient reports exertional chest pressure, dyspnea on exertion, and family history of MI.\n"
    )
    user_parts = []
    for i, ex in enumerate(examples):
        user_parts.append(f"### Example {i+1}")
        user_parts.append(f"Conversation:\n{ex['conversation'].strip()}")
        ex_ksoap = ex['ksoap'].strip()
        cut = ex_ksoap.find("[Disclaimer]")
        if cut != -1:
            ex_ksoap = ex_ksoap[:cut].strip()
        user_parts.append(f"K-SOAP Note (body only):\n{ex_ksoap}")
        user_parts.append("")
    user_parts.append("### Now generate the K-SOAP note BODY ONLY for:")
    user_parts.append(f"Clinical Keywords (copy verbatim into K):\n{keyword_section}")
    user_parts.append(f"\nFact Table:\n{fact_table}")
    user_parts.append("\nK-SOAP Note (STOP after P: line. NO AI Suggestion here):\nK:")
    return system, "\n".join(user_parts)


def _parse_ksoap_sections(raw_output):
    cleaned = raw_output
    cleaned = re.sub(r'\*\*([A-Z]):\*\*', r'\1:', cleaned)
    cleaned = re.sub(r'\*\*([A-Z])\*\*\s*:', r'\1:', cleaned)
    cleaned = re.sub(r'\*\*', '', cleaned)
    cleaned = re.sub(r'#{1,4}\s*', '', cleaned)
    sections = {"K": "", "S": "", "O": "", "A": "", "P": ""}
    pattern  = re.compile(r'(?:^|\n)\s*([KSOAP])\s*:\s*', re.MULTILINE)
    matches  = list(pattern.finditer(cleaned))
    for i, m in enumerate(matches):
        key   = m.group(1)
        start = m.end()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned)
        content = cleaned[start:end].strip()
        if key in sections and not sections[key]:
            sections[key] = content
    return sections


# ── SOAPAgent ─────────────────────────────────────────────────────────────────

class SOAPAgent:
    """Generates SOAP note from transcript and coded entities."""

    def __init__(self):
        self._fewshot_embs = None

    def _get_fewshot_embs(self):
        if self._fewshot_embs is None:
            self._fewshot_embs = _build_fewshot_embeddings()
        return self._fewshot_embs

    def run(self, source: str, entities: list = None, k_shot: int = 3,
            strategy: str = "query_aware"):
        if k_shot == 0:
            return self._run_baseline(source, entities)
        return self._run_ksoap(source, entities, k_shot, strategy)

    def _run_baseline(self, source: str, entities: list = None):
        print("  [SOAPAgent] Generating baseline SOAP note...")
        t0 = time.time()

        if entities:
            fact_table, facts = build_fact_table(entities, source)
        else:
            facts = {k: [] for k in ("confirmed_symptoms","negated_symptoms","diseases",
                                     "family_history","historical","medications_current",
                                     "medications_denied","allergies","social_history",
                                     "vitals_and_exam","procedures_ordered","procedures_denied",
                                     "unclassified")}
            fact_table = f"TRANSCRIPT (no fact table available):\n{source[:2000]}"

        if USE_SELF_CONSISTENCY and entities:
            n = SELF_CONSISTENCY_N
            print(f"  [SOAPAgent] Self-consistency: generating {n} candidates...")
            cands = []
            for k in range(n):
                soap_k, _, score, halluc, cov = _generate_soap_body_once(fact_table, facts, seed=42 + k * 7)
                cands.append((score, halluc, cov, soap_k))
            cands.sort(key=lambda x: x[0])
            score, halluc, cov, soap = cands[0]
        else:
            soap, _, score, halluc, cov = _generate_soap_body_once(fact_table, facts)

        for _k in "SOAP":
            if soap.get(_k):
                soap[_k] = sanitize_assessment(soap[_k])

        soap_body_text  = "\n\n".join(f"{k}: {soap[k]}" for k in "SOAP" if soap[k])
        ai_suggestion_body = _generate_ai_suggestion(soap_body_text)
        ai_suggestion   = f"[Disclaimer]\nAI Suggestion (clinical decision support; NOT part of the SOAP note above):\n{ai_suggestion_body}"
        soap["AI Suggestion"] = ai_suggestion_body
        if entities:
            soap["fact_table"] = fact_table

        soap_raw_parts = [f"{k}: {soap[k]}" for k in "SOAP" if soap[k]]
        soap_raw_parts.append("")
        soap_raw_parts.append(ai_suggestion)
        soap_raw = "\n\n".join(soap_raw_parts).strip()

        print(f"  [SOAPAgent] Done in {round(time.time()-t0,1)}s")
        return soap, soap_raw

    def _run_ksoap(self, source: str, entities: list, k_shot: int, strategy: str):
        print(f"  [SOAPAgent] Generating {k_shot}-shot K-SOAP note (strategy={strategy})...")
        t0 = time.time()

        fact_table, facts   = build_fact_table(entities, source)
        keyword_section     = build_keyword_section(entities)
        fewshot_embs        = self._get_fewshot_embs()

        if strategy == "smallest":
            indices = _select_fewshot_smallest(fewshot_embs, k=k_shot)
        else:
            indices = _select_fewshot_query_aware(fewshot_embs, source[:500], k=k_shot)

        examples = [FEWSHOT_BANK[i] for i in indices]
        specs    = [ex["specialty"] for ex in examples]
        print(f"  [SOAPAgent] Selected {len(examples)} examples: {specs}")

        system_msg, user_msg = _build_ksoap_body_prompt(examples, fact_table, keyword_section)
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ]

        if USE_SELF_CONSISTENCY:
            n = SELF_CONSISTENCY_N
            print(f"  [SOAPAgent] Self-consistency: generating {n} candidates...")
            cands = []
            for k in range(n):
                raw  = _llm(messages, max_tokens=1300, temperature=0.1, seed=100 + k * 13)
                secs = _parse_ksoap_sections(raw)
                if not secs["K"]:
                    secs["K"] = keyword_section
                secs = _hard_default_sections(secs, facts)
                score, halluc, cov = _score_soap_candidate(secs, facts)
                cands.append((score, halluc, cov, secs, raw))
            cands.sort(key=lambda x: x[0])
            _, halluc, cov, sections, raw = cands[0]
        else:
            raw      = _llm(messages, max_tokens=1300, temperature=0.1)
            sections = _parse_ksoap_sections(raw)
            if not sections["K"]:
                sections["K"] = keyword_section
            sections = _hard_default_sections(sections, facts)

        for _k in "SOAP":
            if sections.get(_k):
                sections[_k] = sanitize_assessment(sections[_k])

        body_text          = "\n\n".join(f"{k}: {sections[k]}" for k in "KSOAP" if sections[k])
        ai_suggestion_body = _generate_ai_suggestion(body_text)
        ai_suggestion      = f"[Disclaimer]\nAI Suggestion (clinical decision support; NOT part of the K-SOAP note above):\n{ai_suggestion_body}"

        sections["AI Suggestion"] = ai_suggestion_body
        sections["fact_table"]    = fact_table

        ksoap_raw_parts = [f"{k}: {sections[k]}" for k in "KSOAP" if sections[k]]
        ksoap_raw_parts.append("")
        ksoap_raw_parts.append(ai_suggestion)
        ksoap_raw = "\n\n".join(ksoap_raw_parts).strip()

        print(f"  [SOAPAgent] Done in {round(time.time()-t0,1)}s")
        return sections, ksoap_raw
