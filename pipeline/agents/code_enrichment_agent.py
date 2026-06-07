import re
import json
import textwrap
import torch
from sentence_transformers import util
import pipeline.models as models
from pipeline.utils import _llm, _parse_json
from pipeline.config import ICD10_COSINE_THRESHOLD, RERANK_THRESHOLD, SNOMED_COSINE_THRESHOLD

# ── SNOMED table (exact from notebook Cell 9) ─────────────────────────────────

SNOMED_TABLE = [
    {"id":"422587007","t":"Nausea","c":"R11.0","kw":["nausea","nauseated","nauseous","feeling sick"]},
    {"id":"422400008","t":"Vomiting","c":"R11.10","kw":["vomiting","vomit","throw up","throwing up","emesis"]},
    {"id":"73879007","t":"Nausea and vomiting","c":"R11.2","kw":["nausea and vomiting","nausea with vomiting"]},
    {"id":"62315008","t":"Diarrhea","c":"R19.7","kw":["diarrhea","diarrhoea","loose stools","watery stools","diadea"]},
    {"id":"14760008","t":"Constipation","c":"K59.00","kw":["constipation","constipated"]},
    {"id":"21522001","t":"Abdominal pain","c":"R10.9","kw":["abdominal pain","belly pain","stomach pain","stomachache","abdominal tenderness"]},
    {"id":"73063007","t":"Abdominal cramps","c":"R10.84","kw":["cramps","crampy","cramping","abdominal cramps","stomach cramps"]},
    {"id":"698065002","t":"Heartburn","c":"R12","kw":["heartburn","acid reflux","pyrosis"]},
    {"id":"235365009","t":"GERD","c":"K21.0","kw":["gerd","gastroesophageal reflux","reflux disease"]},
    {"id":"235856003","t":"Gastroenteritis","c":"K52.9","kw":["gastroenteritis","stomach flu","stomach bug","food poisoning"]},
    {"id":"4556007","t":"Gastritis","c":"K29.70","kw":["gastritis"]},
    {"id":"64226004","t":"Dysphagia","c":"R13.10","kw":["dysphagia","difficulty swallowing","trouble swallowing"]},
    {"id":"271681002","t":"Loss of appetite","c":"R63.0","kw":["loss of appetite","not eating","not hungry","decreased appetite","anorexia"]},
    {"id":"249497008","t":"Bloating","c":"R14.0","kw":["bloating","bloated","distended","abdominal distension"]},
    {"id":"162607003","t":"Blood in stool","c":"K92.1","kw":["blood in stool","blood in stools","blood in your stools","rectal bleeding","bloody stool","melena","hematochezia"]},
    {"id":"422823003","t":"Hematemesis","c":"K92.0","kw":["blood in vomit","hematemesis","vomiting blood"]},
    {"id":"197456007","t":"IBS","c":"K58.9","kw":["irritable bowel","ibs"]},
    {"id":"396332003","t":"Appendicitis","c":"K35.80","kw":["appendicitis"]},
    {"id":"34000006","t":"Crohn disease","c":"K50.90","kw":["crohn","crohn's"]},
    {"id":"64766004","t":"Ulcerative colitis","c":"K51.90","kw":["ulcerative colitis","colitis"]},
    {"id":"60728008","t":"Jaundice","c":"R17","kw":["jaundice","yellow skin","icterus","yellow eyes"]},
    {"id":"29857009","t":"Chest pain","c":"R07.9","kw":["chest pain","chest tightness","tight feeling in my chest","pressure on my chest","chest pressure"]},
    {"id":"80313002","t":"Palpitations","c":"R00.2","kw":["palpitations","heart racing","racing of the heart","heart pounding","beating faster","heart fluttering","thumping"]},
    {"id":"3424008","t":"Tachycardia","c":"R00.0","kw":["tachycardia","fast heart rate","rapid heart rate"]},
    {"id":"48867003","t":"Bradycardia","c":"R00.1","kw":["bradycardia","slow heart rate"]},
    {"id":"42343007","t":"Heart failure","c":"I50.9","kw":["heart failure","chf","congestive heart failure","fluid overload"]},
    {"id":"22298006","t":"Myocardial infarction","c":"I21.9","kw":["heart attack","myocardial infarction","mi","stemi","nstemi"]},
    {"id":"194828000","t":"Angina","c":"I20.9","kw":["angina","angina pectoris"]},
    {"id":"38341003","t":"Hypertension","c":"I10","kw":["high blood pressure","hypertension","elevated bp","elevated blood pressure","blood pressure high"]},
    {"id":"45007003","t":"Hypotension","c":"I95.9","kw":["low blood pressure","hypotension"]},
    {"id":"49436004","t":"Atrial fibrillation","c":"I48.91","kw":["atrial fibrillation","afib","a-fib","irregular heartbeat"]},
    {"id":"233604007","t":"Pericarditis","c":"I30.9","kw":["pericarditis","pericardial"]},
    {"id":"253273006","t":"Heart murmur","c":"R01.1","kw":["heart murmur","murmur"]},
    {"id":"59282003","t":"Pulmonary edema","c":"J81.0","kw":["fluid in lungs","pulmonary edema","fluid in the lungs"]},
    {"id":"49727002","t":"Cough","c":"R05.9","kw":["cough","coughing","persistent cough","dry cough","productive cough"]},
    {"id":"267036007","t":"Dyspnea","c":"R06.00","kw":["shortness of breath","difficulty breathing","trouble breathing","short of breath","breathless","dyspnea","winded","gasping for air"]},
    {"id":"56018004","t":"Wheezing","c":"R06.2","kw":["wheeze","wheezing"]},
    {"id":"48409008","t":"Orthopnea","c":"R06.01","kw":["orthopnea","can't breathe lying flat","breathe lying flat","prop up with pillows","pillows at night"]},
    {"id":"70572006","t":"Stridor","c":"R06.1","kw":["stridor","noisy breathing"]},
    {"id":"195967001","t":"Asthma","c":"J45.909","kw":["asthma","asthmatic","reactive airway"]},
    {"id":"13645005","t":"COPD","c":"J44.1","kw":["copd","chronic obstructive","emphysema"]},
    {"id":"233703007","t":"Pulmonary embolism","c":"I26.99","kw":["pulmonary embolism","pe","blood clot in lung"]},
    {"id":"18165001","t":"Pleural effusion","c":"J90","kw":["pleural effusion","fluid around lungs"]},
    {"id":"65124004","t":"Crackles","c":"R09.89","kw":["crackles","rales","crepitations","bibasilar crackles"]},
    {"id":"267038008","t":"Edema","c":"R60.0","kw":["swelling","edema","oedema","swollen","pitting edema","ankle swelling","swelling in my ankles","swelling in ankles","leg swelling"]},
    {"id":"87433001","t":"Hemoptysis","c":"R04.2","kw":["coughing blood","hemoptysis","blood in sputum","bringing up blood"]},
    {"id":"68235000","t":"Nasal congestion","c":"R09.81","kw":["congestion","stuffy nose","blocked nose","runny nose","nasal congestion","rhinorrhea"]},
    {"id":"162397003","t":"Sore throat","c":"R07.0","kw":["sore throat","throat pain","pharyngitis","scratchy throat"]},
    {"id":"301354004","t":"Mucus production","c":"R09.3","kw":["mucus","phlegm","sputum","clear mucus"]},
    {"id":"25064002","t":"Headache","c":"R51.9","kw":["headache","headaches","head pain","cephalgia"]},
    {"id":"37796009","t":"Migraine","c":"G43.909","kw":["migraine","migraines"]},
    {"id":"404640003","t":"Dizziness","c":"R42","kw":["dizziness","dizzy","lightheaded","lightheadedness","vertigo","room spinning"]},
    {"id":"271594007","t":"Syncope","c":"R55","kw":["syncope","fainted","fainting","faint","passed out","loss of consciousness","lost consciousness","blacked out"]},
    {"id":"62106007","t":"Numbness","c":"R20.0","kw":["numbness","numb","tingling","paresthesia","pins and needles"]},
    {"id":"26079004","t":"Tremor","c":"R25.1","kw":["tremor","tremors","shaking","shaky"]},
    {"id":"91175000","t":"Seizure","c":"R56.9","kw":["seizure","seizures","convulsion","convulsions","fit","fits"]},
    {"id":"230690007","t":"Stroke","c":"I63.9","kw":["stroke","cva","cerebrovascular"]},
    {"id":"52448006","t":"Dementia","c":"F03.90","kw":["dementia","alzheimer","memory loss","cognitive decline"]},
    {"id":"23056005","t":"Sciatica","c":"M54.30","kw":["sciatica","sciatic","shooting pain down leg"]},
    {"id":"128196005","t":"Confusion","c":"R41.0","kw":["confusion","confused","disoriented","altered mental status"]},
    {"id":"40917007","t":"Visual disturbance","c":"H53.9","kw":["blurry vision","blurred vision","double vision","diplopia","visual changes"]},
    {"id":"161891005","t":"Back pain","c":"M54.9","kw":["back pain","backache","lower back pain","back is killing","lumbago"]},
    {"id":"57676002","t":"Joint pain","c":"M25.50","kw":["joint pain","arthralgia","knee pain","hip pain","shoulder pain","elbow pain","wrist pain"]},
    {"id":"68962001","t":"Muscle pain","c":"M79.10","kw":["muscle aches","muscle pain","myalgia","body aches","sore muscles"]},
    {"id":"55300003","t":"Muscle cramp","c":"R25.2","kw":["muscle cramp","muscle spasm","charley horse"]},
    {"id":"263171000","t":"Stiffness","c":"M25.60","kw":["stiffness","stiff","joint stiffness"]},
    {"id":"125605004","t":"Fracture","c":"T14.8XXA","kw":["fracture","broken bone","broken"]},
    {"id":"44465007","t":"Sprain","c":"S93.409A","kw":["sprain","sprained","rolled ankle","twisted","ankle injury","ankle sprain"]},
    {"id":"299308007","t":"Neck pain","c":"M54.2","kw":["neck pain","stiff neck","neck stiffness"]},
    {"id":"73211009","t":"Diabetes mellitus","c":"E11.9","kw":["diabetes","diabetic","type 2 diabetes","type 1 diabetes","sugar","blood sugar"]},
    {"id":"40930008","t":"Hypothyroidism","c":"E03.9","kw":["hypothyroidism","underactive thyroid","low thyroid"]},
    {"id":"34095006","t":"Dehydration","c":"E86.0","kw":["dehydration","dehydrated"]},
    {"id":"190372001","t":"Polydipsia","c":"R63.1","kw":["polydipsia","excessive thirst","very thirsty","thirsty","increased thirst"]},
    {"id":"55822004","t":"Hyperlipidemia","c":"E78.5","kw":["high cholesterol","cholesterol","cholesterol problems","hyperlipidemia","dyslipidemia","triglycerides"]},
    {"id":"5291005","t":"Hypoglycemia","c":"E16.2","kw":["hypoglycemia","low blood sugar","low sugar"]},
    {"id":"190268003","t":"Obesity","c":"E66.9","kw":["obesity","obese","overweight","bmi"]},
    {"id":"238131007","t":"Weight loss","c":"R63.4","kw":["weight loss","losing weight","lost weight","unintentional weight loss"]},
    {"id":"8943002","t":"Weight gain","c":"R63.5","kw":["weight gain","gaining weight","gained weight"]},
    {"id":"162116003","t":"Urinary frequency","c":"R35.0","kw":["urinary frequency","frequent urination","peeing a lot","pee a lot","going to bathroom often"]},
    {"id":"49650001","t":"Dysuria","c":"R30.0","kw":["dysuria","pain when peeing","painful urination","burning urination","pain when urinating"]},
    {"id":"75088002","t":"Urinary urgency","c":"R39.15","kw":["urinary urgency","urgent urination","need to go urgently"]},
    {"id":"165232002","t":"Urinary incontinence","c":"R32","kw":["incontinence","urine leakage","leaking urine","can't hold it"]},
    {"id":"68566005","t":"UTI","c":"N39.0","kw":["uti","urinary tract infection","bladder infection"]},
    {"id":"34436003","t":"Hematuria","c":"R31.9","kw":["hematuria","blood in urine","bloody urine"]},
    {"id":"236578006","t":"Kidney stones","c":"N20.0","kw":["kidney stone","kidney stones","renal calculus","nephrolithiasis"]},
    {"id":"197927001","t":"Flank pain","c":"R10.9","kw":["flank pain","kidney pain","side pain"]},
    {"id":"77386006","t":"Pregnancy","c":"Z33.1","kw":["pregnancy","pregnant","expecting"]},
    {"id":"14094001","t":"Amenorrhea","c":"N91.2","kw":["amenorrhea","missed period","late period","no period","absent menstruation","period missed","period late","missed periods"]},
    {"id":"289903006","t":"Menstrual finding","c":"N94.89","kw":["period","periods","menstrual","menstruation","menses"]},
    {"id":"266897007","t":"Menstrual irregularity","c":"N92.6","kw":["irregular period","irregular periods","irregular menstruation"]},
    {"id":"418290006","t":"Dysmenorrhea","c":"N94.6","kw":["painful period","period pain","menstrual cramps","menstrual pain","dysmenorrhea"]},
    {"id":"237079002","t":"Morning sickness","c":"O21.0","kw":["morning sickness"]},
    {"id":"169553002","t":"Contraception","c":"Z30.9","kw":["birth control","oral contraceptive","contraception","contraceptive","condom","condoms","iud"]},
    {"id":"386661006","t":"Fever","c":"R50.9","kw":["fever","fevers","febrile","temperature elevated","hot today","feel hot","feeling hot","warm at night"]},
    {"id":"43724002","t":"Chills","c":"R68.83","kw":["chills","shivering","rigors"]},
    {"id":"84229001","t":"Fatigue","c":"R53.83","kw":["fatigue","tired","exhausted","exhaustion","malaise","lethargy","low energy","feeling tired"]},
    {"id":"42984000","t":"Night sweats","c":"R61","kw":["night sweats","sweating at night","sweats"]},
    {"id":"271807003","t":"Rash","c":"R21","kw":["rash","rashes","skin rash","eruption","hives","urticaria"]},
    {"id":"197480006","t":"Anxiety","c":"F41.9","kw":["anxiety","anxious","worried","worry","panic","panic attack","nervousness"]},
    {"id":"35489007","t":"Depression","c":"F32.9","kw":["depression","depressed","feeling down","low mood","sadness"]},
    {"id":"193462001","t":"Insomnia","c":"G47.00","kw":["insomnia","can't sleep","difficulty sleeping","trouble sleeping","sleep problems"]},
    {"id":"6142004","t":"Influenza","c":"J11.1","kw":["flu","influenza","flu-like symptoms","flu-like"]},
    {"id":"186747009","t":"COVID-19","c":"U07.1","kw":["covid","covid-19","coronavirus","sars-cov-2"]},
    {"id":"36971009","t":"Sinusitis","c":"J32.9","kw":["sinusitis","sinus infection","sinus"]},
    {"id":"55735004","t":"Bronchitis","c":"J20.9","kw":["bronchitis"]},
    {"id":"302911003","t":"Pharyngitis","c":"J02.9","kw":["pharyngitis","strep throat"]},
    {"id":"419199007","t":"Allergy","c":"T78.40XA","kw":["allergy","allergic","allergies"]},
    {"id":"91936005","t":"Penicillin allergy","c":"Z88.0","kw":["penicillin allergy","allergic to penicillin"]},
    {"id":"271737000","t":"Anemia","c":"D64.9","kw":["anemia","anaemia","low hemoglobin","low iron"]},
    {"id":"363346000","t":"Cancer","c":"C80.1","kw":["cancer","malignancy","carcinoma","tumor","tumour","neoplasm"]},
    {"id":"60862001","t":"Tinnitus","c":"H93.19","kw":["tinnitus","ringing in ears","ringing ears"]},
    {"id":"412244003","t":"Ginger preparation","c":None,"kw":["ginger","ginger things","ginger supplement","ginger chews","ginger candy"]},
    {"id":"372614000","t":"Lisinopril","c":None,"kw":["lisinopril"]},
    {"id":"386864001","t":"Amlodipine","c":None,"kw":["amlodipine","norvasc"]},
    {"id":"109081006","t":"Metformin","c":None,"kw":["metformin","glucophage"]},
    {"id":"372826007","t":"Metoprolol","c":None,"kw":["metoprolol","lopressor","toprol"]},
    {"id":"387475002","t":"Furosemide","c":None,"kw":["furosemide","lasix"]},
    {"id":"387207008","t":"Naproxen","c":None,"kw":["naproxen","aleve","naprosyn"]},
    {"id":"386845007","t":"Gabapentin","c":None,"kw":["gabapentin","neurontin"]},
    {"id":"412588001","t":"Rosuvastatin","c":None,"kw":["rosuvastatin","rosvastetin","crestor"]},
    {"id":"373462001","t":"Atorvastatin","c":None,"kw":["atorvastatin","lipitor"]},
    {"id":"387517004","t":"Paracetamol","c":None,"kw":["acetaminophen","tylenol","paracetamol"]},
    {"id":"387458008","t":"Aspirin","c":None,"kw":["aspirin","asa"]},
    {"id":"764146007","t":"Penicillin","c":None,"kw":["penicillin"]},
    {"id":"387501005","t":"Amoxicillin","c":None,"kw":["amoxicillin","amoxil"]},
    {"id":"387322009","t":"Multivitamin","c":None,"kw":["multivitamin","vitamin","vitamins"]},
    {"id":"108774000","t":"Losartan","c":None,"kw":["losartan","cozaar"]},
    {"id":"387207008","t":"Albuterol","c":None,"kw":["albuterol","ventolin","salbutamol","inhaler","puffer","puffers","nebulizer"]},
    {"id":"126218002","t":"Omeprazole","c":None,"kw":["omeprazole","prilosec"]},
    {"id":"387168006","t":"Levothyroxine","c":None,"kw":["levothyroxine","synthroid"]},
    {"id":"387286002","t":"Prednisone","c":None,"kw":["prednisone","prednisolone","steroid","steroids"]},
    {"id":"372756006","t":"Warfarin","c":None,"kw":["warfarin","coumadin","blood thinner"]},
    {"id":"229880002","t":"Cannabis use","c":None,"kw":["marijuana","cannabis","weed","pot"]},
    {"id":"77176002","t":"Smoking","c":"F17.210","kw":["smoking","smoker","cigarette","cigarettes","tobacco","pack a day","half a pack"]},
    {"id":"228273003","t":"Alcohol use","c":"F10.10","kw":["alcohol","drinking","drinks","wine","beer","liquor"]},
    {"id":"399208008","t":"Chest X-ray","c":None,"kw":["chest x-ray","cxr","chest xray","x-ray"]},
    {"id":"113091000","t":"MRI","c":None,"kw":["mri","magnetic resonance","lumbar mri","cervical mri","brain mri"]},
    {"id":"77477000","t":"CT scan","c":None,"kw":["ct scan","ct","cat scan","computed tomography"]},
    {"id":"16310003","t":"Ultrasound","c":None,"kw":["ultrasound","ultrasonography","sonogram","echo","echocardiogram"]},
    {"id":"29303009","t":"ECG","c":None,"kw":["ecg","ekg","electrocardiogram"]},
    {"id":"26604007","t":"CBC","c":None,"kw":["complete blood count","cbc","blood count","blood work"]},
    {"id":"20109005","t":"BMP","c":None,"kw":["basic metabolic panel","bmp","metabolic panel"]},
    {"id":"167252002","t":"Pregnancy test","c":None,"kw":["pregnancy test","urine pregnancy","hcg","beta hcg"]},
    {"id":"27171005","t":"Urinalysis","c":None,"kw":["urinalysis","urine test","urine analysis","ua"]},
    {"id":"183524004","t":"Referral","c":None,"kw":["referral","referring","consultation","cardiology referral","orthopedic referral","specialist","cardiology","neurology","orthopedic"]},
    {"id":"75367002","t":"Blood pressure","c":"R03.0","kw":["blood pressure","bp","systolic","diastolic"]},
    {"id":"386725007","t":"Body temperature","c":"R50.9","kw":["temperature","temp","38 degrees","39 degrees","100.4","101","102","103","104"]},
    {"id":"431314004","t":"Oxygen saturation","c":None,"kw":["oxygen saturation","spo2","o2 sat","91 percent","92 percent","88 percent"]},
    {"id":"12063002","t":"Pitting edema","c":"R60.0","kw":["pitting edema","2+ edema","3+ edema","pitting"]},
    {"id":"160303001","t":"Family history of MI","c":"Z82.49","kw":["father had a heart attack","mother had a heart attack","family heart attack","family history of heart attack"]},
    {"id":"160357008","t":"Family history of cancer","c":"Z80.9","kw":["family cancer","cancer in family","family history of cancer"]},
    {"id":"160377001","t":"Family history of diabetes","c":"Z83.3","kw":["family diabetes","family history of diabetes"]},
]

_AMENORRHEA_CONTEXT = re.compile(
    r"\b(?:missed|late|six\s+weeks?\s+ago|last\s+.*?period|"
    r"weeks?\s+without|not\s+on\s+my\s+period|haven'?t\s+had)\b", re.I
)

def _contextual_period_snomed(entity_text, ctx):
    if not ctx:
        return None
    if _AMENORRHEA_CONTEXT.search(ctx.lower()):
        return {"concept_id":"14094001","term":"Amenorrhea","entity_text":entity_text,
                "method":"context_disambiguated","crosswalk_icd10":"N91.2",
                "crosswalk_icd10_desc":"Amenorrhea, unspecified"}
    return None

def get_snomed(entity_text, transcript_context=""):
    text_lower = entity_text.lower().strip()
    for entry in SNOMED_TABLE:
        matched = False
        for kw in entry["kw"]:
            if len(kw) < 4:
                if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                    matched = True
                    break
            else:
                if kw in text_lower or text_lower in kw:
                    matched = True
                    break
        if matched:
            if entry["t"] == "Menstrual finding":
                period_sn = _contextual_period_snomed(entity_text, transcript_context)
                if period_sn:
                    return period_sn
            result = {
                "concept_id": entry["id"], "term": entry["t"],
                "entity_text": entity_text, "method": "keyword",
            }
            if entry["c"]:
                result["crosswalk_icd10"] = entry["c"]
                import simple_icd_10_cm as icd
                try:
                    result["crosswalk_icd10_desc"] = icd.get_description(entry["c"])
                except Exception:
                    result["crosswalk_icd10_desc"] = ""
            return result

    if not models.embedder:
        return None
    q_emb = models.embedder.encode(entity_text, convert_to_tensor=True)
    snomed_terms = [e["t"] for e in SNOMED_TABLE]
    snomed_embs  = models.embedder.encode(snomed_terms, convert_to_tensor=True)
    hits = util.semantic_search(q_emb, snomed_embs, top_k=3)[0]
    if hits and hits[0]["score"] >= SNOMED_COSINE_THRESHOLD:
        best = SNOMED_TABLE[hits[0]["corpus_id"]]
        result = {
            "concept_id": best["id"], "term": best["t"],
            "entity_text": entity_text, "method": "semantic",
        }
        if best["c"]:
            result["crosswalk_icd10"] = best["c"]
        return result
    return None


# ── ICD-10 (exact from notebook Cell 10) ─────────────────────────────────────

_NORMALIZE_PROMPT = textwrap.dedent("""
You are a medical terminology normalization engine.

Given a list of medical entities extracted from a patient transcript,
convert each one to its STANDARD medical term.

RULES:
1. Convert colloquial patient language to formal medical terminology.
2. Keep the meaning identical — only change the wording.
3. If the entity is already a standard medical term, keep it as-is.
4. Return ONLY a JSON object mapping original → normalized.

EXAMPLES:
Input: ["racing of the heart", "belly pain", "threw up", "can't breathe"]
Output: {"racing of the heart": "palpitations", "belly pain": "abdominal pain", "threw up": "vomiting", "can't breathe": "dyspnea"}

Input: ["chest pain", "headaches", "nausea"]
Output: {"chest pain": "chest pain", "headaches": "headache", "nausea": "nausea"}

Input: ["ginger things", "birth control", "high blood pressure"]
Output: {"ginger things": "ginger supplement", "birth control": "oral contraceptive", "high blood pressure": "hypertension"}

Return ONLY the JSON object. No other text.
""").strip()

_FAST_NORMALIZE = {
    "nauseated": "nausea", "nauseous": "nausea", "feeling sick": "nausea",
    "throw up": "vomiting", "throwing up": "vomiting", "threw up": "vomiting",
    "throwing up blood": "hematemesis", "vomiting blood": "hematemesis",
    "crampy": "abdominal cramps", "belly pain": "abdominal pain",
    "stomach pain": "abdominal pain", "tummy pain": "abdominal pain",
    "loose stools": "diarrhea", "watery stools": "diarrhea", "the runs": "diarrhea",
    "blood in stool": "rectal bleeding", "blood in stools": "rectal bleeding",
    "blood in your stools": "rectal bleeding", "bloody stool": "rectal bleeding",
    "bloating": "abdominal distension", "bloated": "abdominal distension",
    "heartburn": "acid reflux", "acid reflux": "gastroesophageal reflux disease",
    "indigestion": "dyspepsia",
    "trouble swallowing": "dysphagia", "difficulty swallowing": "dysphagia",
    "painful swallowing": "odynophagia",
    "no appetite": "anorexia", "not eating": "decreased appetite",
    "yellow skin": "jaundice", "yellow eyes": "jaundice",
    "high blood pressure": "hypertension", "high bp": "hypertension",
    "blood pressure high": "hypertension",
    "low blood pressure": "hypotension",
    "shortness of breath": "dyspnea", "difficulty breathing": "dyspnea",
    "trouble breathing": "dyspnea", "short of breath": "dyspnea",
    "can't breathe": "dyspnea", "winded": "dyspnea", "gasping for air": "dyspnea",
    "breathless": "dyspnea",
    "chest tightness": "chest pressure", "tight feeling in my chest": "chest pressure",
    "pressure on my chest": "chest pressure", "tight chest": "chest pressure",
    "heart racing": "palpitations", "racing of the heart": "palpitations",
    "beating faster": "tachycardia", "heart pounding": "palpitations",
    "heart fluttering": "palpitations", "irregular heartbeat": "atrial fibrillation",
    "high cholesterol": "hyperlipidemia", "cholesterol problems": "hyperlipidemia",
    "coughing blood": "hemoptysis", "bringing up blood": "hemoptysis",
    "noisy breathing": "stridor", "wheeze": "wheezing",
    "can't breathe lying flat": "orthopnea", "pillows at night": "orthopnea",
    "prop up with pillows": "orthopnea",
    "lightheaded": "dizziness", "dizzy": "dizziness", "room spinning": "vertigo",
    "passing out": "syncope", "fainted": "syncope", "fainting": "syncope",
    "numb": "numbness", "tingling": "paresthesia", "pins and needles": "paresthesia",
    "shooting pain": "neuralgia",
    "feeling down": "depressed mood", "feeling blue": "depressed mood",
    "can't sleep": "insomnia", "trouble sleeping": "insomnia",
    "warm at night": "night sweats", "sweating at night": "night sweats",
    "burning urination": "dysuria", "pain when peeing": "dysuria",
    "blood in urine": "hematuria", "peeing a lot": "polyuria",
    "missed period": "amenorrhea", "late period": "amenorrhea",
    "heavy period": "menorrhagia", "irregular period": "metrorrhagia",
    "sugar": "diabetes mellitus", "high sugar": "diabetes mellitus",
    "thirsty": "polydipsia", "excessive thirst": "polydipsia",
    "tired": "fatigue", "exhausted": "fatigue", "no energy": "fatigue",
    "swelling": "edema", "swollen": "edema", "pitting edema": "edema",
    "muscle aches": "myalgia", "body aches": "myalgia",
    "stiff joints": "joint stiffness",
    "rash": "skin rash", "hives": "urticaria", "itchy skin": "pruritus",
    "runny nose": "rhinorrhea", "stuffy nose": "nasal congestion",
    "sore throat": "pharyngitis", "scratchy throat": "pharyngitis",
    "ringing in ears": "tinnitus",
    "feel sick": "malaise", "off-color": "malaise",
    "ginger things": "ginger supplement", "ginger chews": "ginger supplement",
    "birth control": "oral contraceptive",
    "flu-like symptoms": "influenza-like illness", "flu-like": "influenza-like illness",
    "crystal meth": "methamphetamine", "crystal": "methamphetamine",
    "pack a day": "tobacco use", "half a pack": "tobacco use",
    "sweaty": "diaphoresis",
    "neck swollen": "cervical lymphadenopathy",
    "water pills": "diuretics", "fluid pills": "diuretics",
    "blood thinner": "anticoagulant", "blood thinners": "anticoagulant",
    "inhaler": "bronchodilator inhaler",
}

def _normalize_entities_batch(entities):
    normalized = {}
    needs_llm  = []
    for ent in entities:
        if ent["status"] == "Negated":
            continue
        text = ent["text"].lower().strip()
        if text in _FAST_NORMALIZE:
            normalized[ent["text"]] = _FAST_NORMALIZE[text]
        else:
            needs_llm.append(ent["text"])
    if needs_llm:
        msg = [
            {"role": "system", "content": _NORMALIZE_PROMPT},
            {"role": "user", "content": f"Normalize these: {json.dumps(needs_llm)}\n\nJSON object:"}
        ]
        raw = _llm(msg, max_tokens=500)
        llm_result = _parse_json(raw, "dict")
        if isinstance(llm_result, dict):
            for orig, norm in llm_result.items():
                if isinstance(norm, str) and len(norm) >= 3:
                    normalized[orig] = norm.strip()
    return normalized

def _determine_likely_chapters(entity_text, entity_label):
    text_lower = entity_text.lower()
    chapters = set()
    rules = {
        "K,R": ["nausea","vomit","abdomin","cramp","stomach","belly","diarrhea","constipat","heartburn","bowel","stool","gastri"],
        "N,O,R,Z": ["pregnan","amenorrh","period","menstr","morning sickness","gravid","obstet","contracept"],
        "N,R": ["urin","bladder","kidney","micturi","dysuria","polydipsia"],
        "I,R": ["hypertens","blood pressure","heart","chest pain","palpitat","cardiac","coronary","angina","tachycard","infarct"],
        "J,R": ["cough","breath","dyspnea","wheez","pulmon","lung","asthma","copd","pneumon","bronch","stridor"],
        "M,R": ["muscle","joint","back pain","arthral","myalgia","sciatica","sprain","fracture","stiff"],
        "E,R": ["diabet","thyroid","dehydrat","cholesterol","lipid","polydipsia","obesity","hypoglyc"],
        "L,T": ["allerg","nickel","rash","dermatit","eczema","hives","urticaria","pruritus"],
        "G,R": ["headache","migraine","dizz","seizure","numb","tingling","stroke","tremor","vertigo","syncope"],
        "F":   ["anxiety","depress","insomnia","panic","substance"],
    }
    for chapter_str, terms in rules.items():
        if any(t in text_lower for t in terms):
            chapters.update(chapter_str.split(","))
    if entity_label == "Symptom":
        chapters.add("R")
    if not chapters:
        chapters = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return list(chapters)

def _retrieve_rerank(search_text, entity_label):
    likely_chapters = _determine_likely_chapters(search_text, entity_label)
    q_emb = models.embedder.encode(search_text, convert_to_tensor=True)
    hits  = util.semantic_search(q_emb, models.icd10_embeddings, top_k=25)[0]
    all_cands = []
    for h in hits:
        if h["score"] >= ICD10_COSINE_THRESHOLD:
            idx  = h["corpus_id"]
            code = models.icd10_db.iloc[idx]["code"]
            desc = models.icd10_db.iloc[idx]["description"]
            chapter_bonus = 0.05 if code[0] in likely_chapters else 0.0
            all_cands.append({"code": code, "description": desc,
                               "cosine_score": round(h["score"] + chapter_bonus, 4),
                               "chapter": code[0]})
    if not all_cands:
        return {"code": None, "description": None, "cosine_score": 0.0,
                "rerank_score": 0.0, "method": "no_match"}
    seen = {}
    for c in all_cands:
        if c["code"] not in seen or c["cosine_score"] > seen[c["code"]]["cosine_score"]:
            seen[c["code"]] = c
    cands = list(seen.values())
    pairs  = [(search_text, c["description"]) for c in cands]
    scores = models.cross_encoder.predict(pairs)
    for i, s in enumerate(scores):
        cands[i]["rerank_score"] = round(float(s), 4)
    def sort_key(x):
        return x["rerank_score"] + (0.5 if x["chapter"] in likely_chapters else 0.0)
    cands.sort(key=sort_key, reverse=True)
    best = cands[0]
    if best["rerank_score"] >= RERANK_THRESHOLD:
        best["method"] = "universal_router"
        return best
    return {"code": None, "description": None, "cosine_score": 0.0,
            "rerank_score": 0.0, "method": "below_threshold"}

def get_icd10(entity_text, entity_label="Symptom", snomed_crosswalk=None, normalized_text=None):
    if snomed_crosswalk and snomed_crosswalk.get("crosswalk_icd10"):
        code  = snomed_crosswalk["crosswalk_icd10"]
        match = models.icd10_db[models.icd10_db["code"] == code]
        if not match.empty:
            return {"code": code, "description": match.iloc[0]["description"],
                    "cosine_score": 1.0, "rerank_score": 10.0, "method": "SNOMED_crosswalk"}
        return {"code": code, "description": snomed_crosswalk.get("crosswalk_icd10_desc", ""),
                "cosine_score": 1.0, "rerank_score": 10.0, "method": "SNOMED_crosswalk"}
    search_term = normalized_text if normalized_text else entity_text
    return _retrieve_rerank(search_term, entity_label)


# ── CPT codes (exact from notebook Cell 11) ───────────────────────────────────

CPT_TABLE = [
    {"code": "99213", "description": "Office visit, established patient, low complexity",
     "keywords": ["office visit", "follow-up", "routine visit"]},
    {"code": "99214", "description": "Office visit, established patient, moderate complexity",
     "keywords": ["office visit moderate", "established patient moderate"]},
    {"code": "99215", "description": "Office visit, established patient, high complexity",
     "keywords": ["complex visit", "multiple problems"]},
    {"code": "99202", "description": "Office visit, new patient, low complexity",
     "keywords": ["new patient visit", "initial visit"]},
    {"code": "99281", "description": "Emergency department visit, low severity",
     "keywords": ["emergency department", "ED visit"]},
    {"code": "81025", "description": "Urine pregnancy test",
     "keywords": ["pregnancy test", "urine pregnancy", "HCG", "hcg test"]},
    {"code": "59400", "description": "Routine obstetric care",
     "keywords": ["obstetric care", "prenatal care", "pregnancy care"]},
    {"code": "81001", "description": "Urinalysis, automated with microscopy",
     "keywords": ["urinalysis", "urine test", "urine analysis", "UA"]},
    {"code": "80048", "description": "Basic metabolic panel (BMP)",
     "keywords": ["basic metabolic panel", "BMP", "metabolic panel"]},
    {"code": "85025", "description": "Complete blood count with differential (CBC)",
     "keywords": ["complete blood count", "CBC", "blood count"]},
    {"code": "80061", "description": "Lipid panel",
     "keywords": ["lipid panel", "cholesterol test"]},
    {"code": "84443", "description": "Thyroid stimulating hormone (TSH)",
     "keywords": ["TSH", "thyroid test", "thyroid function"]},
    {"code": "82947", "description": "Glucose, quantitative, blood",
     "keywords": ["blood glucose", "glucose test", "blood sugar"]},
    {"code": "71046", "description": "Chest X-ray, 2 views",
     "keywords": ["chest x-ray", "chest xray", "CXR"]},
    {"code": "74018", "description": "Abdominal X-ray, 1 view",
     "keywords": ["abdominal x-ray", "KUB", "abdominal xray"]},
    {"code": "73610", "description": "X-ray of ankle, 3 views",
     "keywords": ["ankle x-ray", "ankle xray", "left ankle x-ray", "right ankle x-ray"]},
    {"code": "72148", "description": "MRI of lumbar spine without contrast",
     "keywords": ["MRI lumbar", "MRI spine", "lumbar MRI", "lumbar spine"]},
    {"code": "72141", "description": "MRI of cervical spine without contrast",
     "keywords": ["MRI cervical", "cervical MRI", "cervical spine"]},
    {"code": "83880", "description": "B-type natriuretic peptide (BNP)",
     "keywords": ["BNP", "natriuretic peptide", "BNP level"]},
    {"code": "80053", "description": "Comprehensive metabolic panel (CMP)",
     "keywords": ["comprehensive metabolic panel", "CMP"]},
    {"code": "93000", "description": "Electrocardiogram (ECG), 12-lead",
     "keywords": ["ECG", "EKG", "electrocardiogram"]},
    {"code": "94760", "description": "Pulse oximetry, single reading",
     "keywords": ["oxygen saturation", "pulse oximetry", "SpO2"]},
    {"code": "94002", "description": "Ventilation assist and management",
     "keywords": ["supplemental oxygen", "oxygen therapy", "oxygen"]},
    {"code": "99242", "description": "Consultation, moderate complexity",
     "keywords": ["referral", "referring", "consultation", "follow-up",
                  "cardiology", "orthopedic", "spine specialist"]},
]

def get_cpt_deterministic(procedure_text):
    text_lower = procedure_text.lower()
    for entry in CPT_TABLE:
        for kw in entry["keywords"]:
            if len(kw) < 4:
                if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                    return {"code": entry["code"], "description": entry["description"],
                            "justification": f"Keyword match: '{procedure_text}'", "method": "keyword"}
            else:
                if kw.lower() in text_lower:
                    return {"code": entry["code"], "description": entry["description"],
                            "justification": f"Keyword match: '{procedure_text}'", "method": "keyword"}

    cpt_descs = [c["description"] for c in CPT_TABLE]
    cpt_embs  = models.embedder.encode(cpt_descs, convert_to_tensor=True, batch_size=64)
    q = models.embedder.encode(procedure_text, convert_to_tensor=True)
    sims = util.cos_sim(q, cpt_embs)[0]
    top_indices = torch.argsort(sims, descending=True)[:5]
    candidates = []
    for idx in top_indices:
        score = float(sims[idx])
        if score >= 0.50:
            candidates.append({"idx": int(idx), "cosine": round(score, 4),
                                "code": CPT_TABLE[int(idx)]["code"],
                                "desc": CPT_TABLE[int(idx)]["description"]})
    if not candidates:
        return None
    pairs = [(procedure_text, c["desc"]) for c in candidates]
    rerank_scores = models.cross_encoder.predict(pairs)
    for i, s in enumerate(rerank_scores):
        candidates[i]["rerank"] = round(float(s), 4)
    candidates.sort(key=lambda x: x["rerank"], reverse=True)
    best = candidates[0]
    if best["rerank"] >= 0.5:
        return {"code": best["code"], "description": best["desc"],
                "justification": f"Reranked semantic ({best['cosine']:.2f}/{best['rerank']:.2f}): '{procedure_text}'",
                "method": "semantic_reranked"}
    return None


# ── CodeEnrichmentAgent ───────────────────────────────────────────────────────

class CodeEnrichmentAgent:
    """Enriches NER entities with SNOMED CT, ICD-10-CM, and CPT codes."""

    def run(self, entities: list, transcript_text: str = "") -> list:
        print("  [CodeEnrichmentAgent] Normalizing entities (MedCodER Step 1)...")
        norm_map = _normalize_entities_batch(entities)
        for orig, norm in norm_map.items():
            if orig.lower() != norm.lower():
                print(f"         '{orig}' -> '{norm}'")

        result = []
        for ent in entities:
            e = ent.copy()
            if ent["status"] == "Negated" or ent.get("label") == "NULL":
                e["snomed"]   = None
                e["icd10_cm"] = None
                e["normalized"] = None
                result.append(e)
                continue

            normalized = norm_map.get(ent["text"], ent["text"])
            e["normalized"] = normalized if normalized != ent["text"] else None

            context = ""
            offset = ent.get("first_offset")
            if offset and transcript_text:
                start = max(0, offset["start"] - 100)
                end_  = min(len(transcript_text), offset["end"] + 100)
                context = transcript_text[start:end_]

            sn = get_snomed(normalized, transcript_context=context)
            if not sn:
                sn = get_snomed(ent["text"], transcript_context=context)
            e["snomed"] = sn

            if (ent["label"] in ("Disease", "Symptom") and
                    ent["status"] in ("Confirmed", "Historical", "Family_History")):
                e["icd10_cm"] = get_icd10(
                    ent["text"], entity_label=ent["label"],
                    snomed_crosswalk=sn, normalized_text=normalized
                )
            else:
                e["icd10_cm"] = None

            result.append(e)
        return result

    def assign_cpt(self, entities: list) -> list:
        cpt_codes = []
        for ent in entities:
            if ent["label"] == "Procedure" and ent["status"] == "Confirmed":
                norm_text = ent.get("normalized") or ent["text"]
                cpt = get_cpt_deterministic(norm_text)
                if not cpt and norm_text != ent["text"]:
                    cpt = get_cpt_deterministic(ent["text"])
                if cpt:
                    cpt_codes.append(cpt)
        print(f"  [CodeEnrichmentAgent] CPT: {len(cpt_codes)} codes assigned")
        return cpt_codes
