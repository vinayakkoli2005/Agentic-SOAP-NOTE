import torch
import pandas as pd
import warnings
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as hf_pipeline, AutoTokenizer
import transformers
from sentence_transformers import SentenceTransformer, CrossEncoder
import simple_icd_10_cm as icd
from huggingface_hub import login

from pipeline.config import (
    HF_TOKEN, WHISPER_MODEL_ID, LLAMA_MODEL_ID,
    SAPBERT_MODEL_ID, CROSS_ENCODER_ID
)

warnings.filterwarnings("ignore")

# Shared model references — populated once by load_all_models()
whisper_pipe       = None
pipeline_llm       = None
pipeline_8b_instruct = None
tokenizer          = None
terminators        = None
embedder           = None
cross_encoder      = None
icd10_db           = None
icd10_embeddings   = None


def load_all_models():
    global whisper_pipe, pipeline_llm, pipeline_8b_instruct
    global tokenizer, terminators, embedder, cross_encoder
    global icd10_db, icd10_embeddings

    print("=== STARTING BOOT SEQUENCE: DOWNLOADING & LOADING ALL MODELS ===")

    login(token=HF_TOKEN)

    print("\n[1/4] Loading Whisper ASR Model...")
    device     = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        WHISPER_MODEL_ID, torch_dtype=torch_dtype,
        low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)
    whisper_pipe = hf_pipeline(
        "automatic-speech-recognition",
        model=model,
        return_timestamps=True,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    print("\n[2/4] Loading LLaMA 3.1 8B Model...")
    pipeline_8b_instruct = transformers.pipeline(
        "text-generation",
        model=LLAMA_MODEL_ID,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_ID)
    terminators = [
        pipeline_8b_instruct.tokenizer.eos_token_id,
        pipeline_8b_instruct.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    pipeline_llm = pipeline_8b_instruct

    print("\n[3/4] Loading SapBERT & Cross-Encoder...")
    embedder      = SentenceTransformer(SAPBERT_MODEL_ID)
    cross_encoder = CrossEncoder(CROSS_ENCODER_ID)

    print("\n[4/4] Building ICD-10 Embeddings...")
    records = [
        {"code": c, "description": icd.get_description(c)}
        for c in icd.get_all_codes() if icd.is_leaf(c)
    ]
    icd10_db = pd.DataFrame(records)
    icd10_embeddings = embedder.encode(
        icd10_db["description"].tolist(),
        convert_to_tensor=True, show_progress_bar=True, batch_size=256,
    )

    print("\n=== BOOT COMPLETE ===")
