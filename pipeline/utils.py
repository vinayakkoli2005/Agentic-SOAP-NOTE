import re
import json
import torch
import textwrap
import pipeline.models as models
from pipeline.config import VALID_LABELS, VALID_STATUSES

# ── LLM wrapper ──────────────────────────────────────────────────────────────

def _llm(messages, max_tokens=2048, temperature=0.15, seed=None):
    kwargs = dict(
        max_new_tokens=max_tokens, eos_token_id=models.terminators,
        do_sample=True, temperature=temperature, top_p=0.9,
        repetition_penalty=1.15,
    )
    if seed is not None:
        torch.manual_seed(seed)
    out = models.pipeline_llm(messages, **kwargs)
    return out[0]["generated_text"][-1]["content"]


# ── JSON parser ───────────────────────────────────────────────────────────────

def _parse_json(raw, expect="list"):
    for fence in ("```json", "```"):
        if fence in raw:
            raw = raw.split(fence)[1].split("```")[0].strip()
            break
    if expect == "dict":
        search_order = [("{", "}"), ("[", "]")]
    else:
        search_order = [("[", "]"), ("{", "}")]
    for opener, closer in search_order:
        start = raw.find(opener)
        if start == -1:
            continue
        depth, end = 0, -1
        for i, ch in enumerate(raw[start:], start):
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
            if depth == 0:
                end = i + 1
                break
        if end != -1:
            raw = raw[start:end]
            break
        if opener == "[" and start != -1:
            last_brace = raw.rfind("}")
            if last_brace > start:
                truncated = raw[start:last_brace + 1] + "]"
                try:
                    test = json.loads(truncated, strict=False)
                    if isinstance(test, list):
                        raw = truncated
                        break
                except Exception:
                    last_comma = truncated.rfind(",")
                    if last_comma > start:
                        truncated2 = truncated[:last_comma] + "]"
                        try:
                            test2 = json.loads(truncated2, strict=False)
                            if isinstance(test2, list):
                                raw = truncated2
                                break
                        except Exception:
                            pass
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    _ESC = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}
    result, in_str, esc_next = [], False, False
    for ch in raw:
        if esc_next:
            result.append(ch)
            esc_next = False
            continue
        if ch == "\\":
            result.append(ch)
            esc_next = in_str
            continue
        if ch == '"':
            in_str = not in_str
        if in_str and ch in _ESC:
            result.append(_ESC[ch])
            continue
        result.append(ch)
    cleaned = "".join(result)
    try:
        parsed = json.loads(cleaned, strict=False)
    except Exception:
        for m in re.finditer(r'[\[{]', cleaned):
            try:
                parsed = json.loads(cleaned[m.start():], strict=False)
                if expect == "list" and isinstance(parsed, list):
                    return parsed
                if expect == "dict" and isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
        return [] if expect == "list" else {}
    if expect == "list":
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    return v
        return []
    else:
        if isinstance(parsed, dict):
            return parsed
        return {}


# ── Schema enforcement ────────────────────────────────────────────────────────

def _enforce_schema(candidates):
    clean, rejected = [], []
    for ent in candidates:
        if not isinstance(ent, dict):
            rejected.append({"raw": ent, "_reason": "not a dict"})
            continue
        text  = ent.get("text", "").strip()
        label = ent.get("label", "").strip()
        status = ent.get("status", "").strip()
        if not text:
            rejected.append({**ent, "_reason": "empty text"})
            continue
        if label not in VALID_LABELS:
            rejected.append({**ent, "_reason": f"invalid label: {label}"})
            continue
        if status not in VALID_STATUSES:
            rejected.append({**ent, "_reason": f"invalid status: {status}"})
            continue
        clean.append({"text": text, "label": label, "status": status,
                      "reasoning": ent.get("reasoning", "")})
    return clean, rejected


# ── SPELL offset verification ─────────────────────────────────────────────────

def _spell_verify_and_align(entity_text, source):
    idx = source.find(entity_text)
    if idx != -1:
        return {"start": idx, "end": idx + len(entity_text)}
    lower_src  = source.lower()
    lower_ent  = entity_text.lower()
    idx = lower_src.find(lower_ent)
    if idx != -1:
        return {"start": idx, "end": idx + len(entity_text)}
    words = entity_text.lower().split()
    if len(words) <= 1:
        return None
    first_word = words[0]
    for m in re.finditer(re.escape(first_word), lower_src):
        segment = lower_src[m.start(): m.start() + len(lower_ent) + 10]
        if all(w in segment for w in words):
            return {"start": m.start(), "end": m.start() + len(entity_text)}
    return None


# ── Negation detection ────────────────────────────────────────────────────────

_NEGATION_PATTERNS = [
    re.compile(r'\b(?:no|not|without|denies|denied|deny|never|negative for|'
               r'absent|absence of|rules? out|ruled out|unlikely|low suspicion for)\b\s+'
               r'(?:\w+\s+){0,4}', re.I),
    re.compile(r'\b(?:no|not)\s+(?:any\s+)?', re.I),
]

_AFFIRMATIVE_QUALIFIERS = re.compile(
    r'\b(?:just|only|a little|slightly|some|occasional|mild|moderate|'
    r'from|due to|because of|secondary to|related to)\b', re.I
)

def _triple_negation(entity_text, source, current_status, label, spell_offset=None):
    if current_status in ("Family_History", "Historical"):
        return current_status
    if spell_offset:
        start = spell_offset.get("start", 0)
        window_start = max(0, start - 80)
        window = source[window_start: start + len(entity_text) + 20]
    else:
        idx = source.lower().find(entity_text.lower())
        if idx == -1:
            return current_status
        window_start = max(0, idx - 80)
        window = source[window_start: idx + len(entity_text) + 20]

    if _AFFIRMATIVE_QUALIFIERS.search(window[:window.lower().find(entity_text.lower()) + 5] if entity_text.lower() in window.lower() else window):
        return "Confirmed"

    for pat in _NEGATION_PATTERNS:
        if pat.search(window):
            return "Negated"

    return current_status


# ── Non-clinical filter ───────────────────────────────────────────────────────

_MEDICAL_KEYWORDS = {
    "pain","ache","aches","aching","sore","soreness","tender","tenderness",
    "fever","fevers","febrile","chills","sweats","sweaty","sweating",
    "cough","coughing","wheeze","wheezing","breathe","breathing","breath",
    "dyspnea","tachypnea","apnea","gasping","winded","suffocating",
    "nausea","nauseated","nauseous","vomiting","vomit","emesis","retching",
    "diarrhea","constipation","bloating","bloated","flatulence",
    "bleeding","blood","hemorrhage","bruise","bruising","hematoma",
    "swelling","swollen","edema","oedema","pitting",
    "headache","migraine","dizziness","dizzy","lightheaded","vertigo","syncope",
    "numbness","numb","tingling","paresthesia","weakness","paralysis",
    "rash","itching","itchy","hives","urticaria","eczema","lesion","blister",
    "fatigue","tired","exhaustion","malaise","lethargy",
    "insomnia","sleep","drowsy","drowsiness",
    "chest","palpitations","tachycardia","bradycardia","arrhythmia","murmur",
    "cramp","cramps","crampy","cramping","spasm","spasms",
    "stiffness","stiff","swollen","lump","mass","nodule","tumor",
    "discharge","secretion","mucus","phlegm","sputum","congestion",
    "thirsty","thirst","polydipsia","polyuria","oliguria","anuria",
    "appetite","anorexia","weight","obesity","cachexia",
    "anxiety","anxious","depression","depressed","panic","agitation",
    "confusion","delirium","hallucination","tremor","seizure","convulsion",
    "fainted","fainting","unconscious","consciousness","coma",
    "urinary","urination","dysuria","hematuria","incontinence","frequency",
    "constipated","indigestion","dyspepsia","heartburn","reflux","regurgitation",
    "jaundice","icterus","ascites","melena","hematemesis",
    "orthopnea","stridor","cyanosis","clubbing","pallor","diaphoresis",
    "tinnitus","vertigo","photophobia","diplopia","blurry","scotoma",
    "dysphagia","odynophagia","hoarseness","aphonia",
    "edematous","erythema","erythematous","purpura","petechiae",
    "rigidity","guarding","rebound","distension","distended",
    "crackles","rales","rhonchi","friction","gallop",
    "diabetes","diabetic","hypertension","hypotension",
    "asthma","copd","bronchitis","pneumonia","tuberculosis",
    "heart","cardiac","coronary","angina","infarction","myocardial",
    "failure","fibrillation","flutter","stenosis","regurgitation",
    "stroke","ischemic","hemorrhagic","aneurysm","embolism","thrombosis","clot",
    "cancer","carcinoma","melanoma","lymphoma","leukemia","tumor","neoplasm",
    "infection","infectious","sepsis","abscess","cellulitis",
    "arthritis","osteoporosis","fracture","dislocation","sprain","strain",
    "sciatica","radiculopathy","neuropathy","myelopathy",
    "anemia","thrombocytopenia","coagulopathy",
    "hypothyroidism","hyperthyroidism","thyroid","goiter",
    "cirrhosis","hepatitis","pancreatitis","cholecystitis","appendicitis",
    "colitis","diverticulitis","hernia","obstruction","ileus",
    "gastritis","gastroenteritis","enteritis","esophagitis",
    "nephritis","pyelonephritis","nephrolithiasis","cystitis",
    "uti","urinary","prostatitis","bph",
    "eczema","psoriasis","dermatitis","cellulitis","folliculitis",
    "migraine","epilepsy","parkinson","alzheimer","dementia","multiple sclerosis",
    "lupus","fibromyalgia","gout","rheumatoid",
    "hiv","aids","hepatitis","herpes","shingles","influenza","flu",
    "copd","emphysema","fibrosis","sarcoidosis","pleurisy","pleural",
    "pericarditis","myocarditis","endocarditis","cardiomyopathy",
    "dvt","pe","pulmonary","aortic","mitral","tricuspid",
    "cholesterol","hyperlipidemia","dyslipidemia","triglyceride",
    "obesity","bmi","overweight","underweight","malnutrition",
    "allergic","allergy","allergies","anaphylaxis","angioedema",
    "pregnancy","pregnant","prenatal","postpartum","miscarriage","ectopic",
    "amenorrhea","dysmenorrhea","menorrhagia","metrorrhagia","menstrual",
    "period","periods","menstruation","ovulation",
    "preeclampsia","eclampsia","gestational","trimester",
    "contraception","contraceptive",
    "medication","medications","medicine","drug","prescription","prescribed",
    "antibiotic","antiviral","antifungal","antihistamine","analgesic",
    "opioid","nsaid","steroid","corticosteroid","immunosuppressant",
    "insulin","metformin","lisinopril","amlodipine","metoprolol",
    "furosemide","losartan","atorvastatin","rosuvastatin","simvastatin",
    "omeprazole","pantoprazole","gabapentin","pregabalin",
    "amoxicillin","azithromycin","ciprofloxacin","penicillin",
    "acetaminophen","tylenol","ibuprofen","advil","motrin","naproxen",
    "aspirin","warfarin","heparin","eliquis","xarelto",
    "prednisone","albuterol","inhaler","puffer","puffers","nebulizer",
    "levothyroxine","synthroid","multivitamin","supplement",
    "marijuana","cannabis","cocaine","methamphetamine","meth","crystal",
    "heroin","fentanyl","oxycodone","hydrocodone","morphine","codeine",
    "alcohol","nicotine","cigarette","cigarettes","smoking","tobacco","vaping",
    "ginger","vitamin","mineral","probiotic",
    "birth control","oral contraceptive","condom","condoms","iud",
    "x-ray","xray","mri","ct","scan","ultrasound","sonogram","echocardiogram",
    "endoscopy","colonoscopy","bronchoscopy","cystoscopy","arthroscopy",
    "biopsy","aspiration","catheterization","intubation","ventilation",
    "ecg","ekg","electrocardiogram","holter","stress test",
    "surgery","surgical","appendectomy","cholecystectomy","hysterectomy",
    "mastectomy","colectomy","laparoscopy","thoracotomy","craniotomy",
    "transplant","implant","stent","pacemaker","defibrillator",
}

def _is_non_clinical(text):
    text_lower = text.lower().strip()
    words = text_lower.split()
    if any(kw in text_lower for kw in _MEDICAL_KEYWORDS):
        return False
    if any(kw in text_lower for kw in {"smoke","smoking","cigarette","tobacco",
                                        "alcohol","cannabis","marijuana","cocaine",
                                        "meth","heroin","birth control","inhaler",
                                        "ginger","vitamin"}):
        return False
    non_clinical_patterns = [
        r'^\d+$', r'^\d+\s*(mg|ml|kg|lb|cm|mm|bpm|mmhg)$',
        r'^(yes|no|okay|ok|sure|right|well|so|and|but|the|a|an)$',
        r'^(doctor|patient|nurse|physician)$',
        r'^\w{1,2}$',
    ]
    for pat in non_clinical_patterns:
        if re.match(pat, text_lower):
            return True
    return False


# ── VerifyNER-style LLM 2nd pass ──────────────────────────────────────────────

_VERIFY_NER_PROMPT = textwrap.dedent("""
You are a clinical NER quality checker.
Given a list of extracted medical entities and the source transcript, verify each entity.
For each entity, decide if the label and status are correct.
Return ONLY a JSON array of the verified entities (remove incorrect ones, keep correct ones).
Each entity must have: text, label, status.
Labels: Drug | Disease | Symptom | Procedure | NULL
Statuses: Confirmed | Negated | Historical | Family_History
""").strip()

def verify_entities_llm(entities, source, batch_size=15):
    verified, rejected = [], []
    for i in range(0, len(entities), batch_size):
        batch = entities[i:i + batch_size]
        batch_simple = [{"text": e["text"], "label": e["label"], "status": e["status"]}
                        for e in batch]
        msg = [
            {"role": "system", "content": _VERIFY_NER_PROMPT},
            {"role": "user", "content": (
                f"Transcript:\n{source[:3000]}\n\n"
                f"Entities to verify:\n{json.dumps(batch_simple)}\n\n"
                "Return verified entities as JSON array:"
            )}
        ]
        raw = _llm(msg, max_tokens=1500, temperature=0.1)
        result = _parse_json(raw, "list")
        verified_texts = {r["text"] for r in result if isinstance(r, dict) and r.get("text")}
        for ent in batch:
            if ent["text"] in verified_texts:
                verified.append(ent)
            else:
                rejected.append({**ent, "_reason": "VerifyNER: rejected by LLM"})
    return verified, rejected


# ── Per-code LLM verification (MedCodER-style) ────────────────────────────────

_VERIFY_CODE_PROMPT = textwrap.dedent("""
You are a medical coding verifier.
Given an entity, its assigned code, and the transcript context, verify if the code is correct.
Return JSON: {"correct": true/false, "reason": "brief reason"}
""").strip()

def verify_code_llm(entities, code_type, transcript_text):
    for ent in entities:
        code_obj = ent.get(code_type)
        if not code_obj or not code_obj.get("code"):
            continue
        context = ""
        offset = ent.get("first_offset")
        if offset and transcript_text:
            s = max(0, offset["start"] - 100)
            e_ = min(len(transcript_text), offset["end"] + 100)
            context = transcript_text[s:e_]
        msg = [
            {"role": "system", "content": _VERIFY_CODE_PROMPT},
            {"role": "user", "content": (
                f"Entity: {ent['text']} ({ent['label']}, {ent['status']})\n"
                f"Code: {code_obj['code']} — {code_obj.get('description', '')}\n"
                f"Context: {context}\n\nVerify JSON:"
            )}
        ]
        raw = _llm(msg, max_tokens=100, temperature=0.1)
        result = _parse_json(raw, "dict")
        if isinstance(result, dict) and result.get("correct") is False:
            ent[code_type] = None
    return entities


# ── Speculation sanitizer ─────────────────────────────────────────────────────

_SPECULATION_PATTERNS = [
    (re.compile(r'\b(?:suspected|likely|probable|possible|presumed)\s+', re.I), ''),
    (re.compile(r'\bconcerning\s+for\s+', re.I), 'presents with '),
    (re.compile(r'\bsuggestive\s+of\s+', re.I), 'presents with '),
    (re.compile(r'\bconsistent\s+with\s+', re.I), 'presents with '),
    (re.compile(r'\brule\s+out\s+', re.I), 'evaluate for '),
    (re.compile(r'\bdifferential\s+diagnos[ie]s?\s+(?:includes?|:)?\s*', re.I), ''),
    (re.compile(r'\bworkup\s+(?:should\s+)?includes?\s+', re.I), ''),
    (re.compile(r'\bin\s+keeping\s+with\s+', re.I), 'presents with '),
    (re.compile(r'\bmay\s+(?:have|be)\s+', re.I), 'reports '),
    (re.compile(r'\bmight\s+(?:have|be)\s+', re.I), 'reports '),
    (re.compile(r'\bcould\s+(?:have|be)\s+', re.I), 'reports '),
]

def sanitize_assessment(text):
    if not text:
        return text
    out = text
    for pat, repl in _SPECULATION_PATTERNS:
        out = pat.sub(repl, out)
    out = re.sub(r'\s+', ' ', out).strip()
    return out
