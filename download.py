"""Download the three models and the two corpora from Hugging Face (no authentication needed).
Set HF_HOME first if you want the cache somewhere else. Usage: python download.py
"""
from huggingface_hub import snapshot_download, hf_hub_download

MODELS = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "HuggingFaceTB/SmolLM2-1.7B"]
for m in MODELS:
    p = snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"])
    print("model", m, "->", p, flush=True)
p = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset")
print("wikitext2 ->", p, flush=True)
p = hf_hub_download("allenai/c4", "en/c4-validation.00000-of-00008.json.gz", repo_type="dataset")
print("c4 ->", p, flush=True)
