# Pipeline config — exact same thresholds and flags as the notebook

import os
HF_TOKEN = os.environ.get("HF_TOKEN", "")

WHISPER_MODEL_ID   = "openai/whisper-large-v3-turbo"
LLAMA_MODEL_ID     = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
SAPBERT_MODEL_ID   = "pritamdeka/S-PubMedBert-MS-MARCO"
CROSS_ENCODER_ID   = "cross-encoder/ms-marco-MiniLM-L-6-v2"

ICD10_COSINE_THRESHOLD  = 0.65
RERANK_THRESHOLD        = 0.8
SNOMED_COSINE_THRESHOLD = 0.90

VALID_LABELS   = {"Drug", "Disease", "Symptom", "Procedure", "NULL"}
VALID_STATUSES = {"Confirmed", "Negated", "Historical", "Family_History"}

USE_NER_VERIFICATION  = True
USE_CODE_VERIFICATION = False
USE_SELF_CONSISTENCY  = False
SELF_CONSISTENCY_N    = 3
