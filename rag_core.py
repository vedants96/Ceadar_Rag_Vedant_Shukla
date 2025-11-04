import os
import re
import json
import pickle
import faiss
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import time, requests


MODEL_ID = os.environ.get("HF_GEN_MODEL", "Qwen/Qwen2.5-3B-Instruct")
LOCAL_DIR = os.environ.get("HF_LOCAL_MODEL_DIR", "").strip()
USE_LOCAL = bool(int(os.environ.get("TRANSFORMERS_LOCAL_ONLY", "0")))  # 1 to force offline

def _model_source() -> str:
    return str(LOCAL_DIR) if 'USE_LOCAL' in globals() and USE_LOCAL else MODEL_ID

hf_tokenizer = AutoTokenizer.from_pretrained(
    _model_source(),
    local_files_only=USE_LOCAL)

def token_len(text: str) -> int:
    return len(hf_tokenizer.encode(text or ""))


def cite_str(row) -> str:
    ps, pe = int(row["page_start"]), int(row["page_end"])
    pages = f"p.{ps}" if ps == pe else f"p.{ps}-{pe}"
    kind = "table" if row.get("content_type") == "table" else "text"
    return f"[{row['doc_label']}: {pages} | {kind}]"


# Context assembly
_HEADER_LINE = re.compile(
    r"^\s*\[[A-Z]:\s*p\.\d+(?:-\d+)?(?:\s*\|\s*(?:text|table))?\]\s*\n",
    re.IGNORECASE,
)

def assemble_context(rows: pd.DataFrame, max_ctx_tokens: int = 2200) -> str:
    # Build a compact evidence block under a strict token budget (for your gen model)

    parts = []                     # collected "[cite]\n<body>\n" blocks
    used  = 0                      # tokens already placed in 'parts'
    seen  = set()                  # optional de-dup by chunk_id if present

    for _, r in rows.iterrows():
        # ---- de-duplicate chunks by 'chunk_id' if available ----
        key = r.get("chunk_id")
        if key and key in seen:
            continue
        if key:
            seen.add(key)

        # ---- build the header line for this chunk ----
        head = r.get("citation") or cite_str(r)      # reuse your citation formatter

        # ---- get the body text (prefer raw 'text'; fallback to 'embedding_text' body) ----
        body = (r.get("text") or "").strip()
        if not body:
            et = str(r.get("embedding_text") or "")
            # strip exactly one leading header line if present
            body = _HEADER_LINE.sub("", et, count=1).strip()

        # ---- prepare the block and count tokens using your tokenizer ----
        block = f"{head}\n{body}\n"
        need  = token_len(block)

        # ---- enforce the global token budget; keep at least one block ----
        if used + need > max_ctx_tokens:
            if not parts:
                parts.append(block)
            break

        # ---- accept this block and update the budget ----
        parts.append(block)
        used += need

    # ---- join blocks into the final context string ----
    return "\n".join(parts).strip()

# Promt Building
SYSTEM_PROMPT = (
    "You are a careful scientific assistant. Answer ONLY using the provided context.\n"
    "If the answer is not in the context, say \"I don't find this in the provided papers.\".\n"
    "Cite every claim with bracket citations like [A: p.12] or [B: p.7-8].\n"
    "Prefer quoting numbers from tables verbatim; do not guess."
)

def build_prompt(question: str, context_block: str) -> str:
    # Composing the final prompt sent to the LLM.
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Answer (with citations):"
    )

_CITE_PATTERN = re.compile(r"\[[A-Z]:\s*p\.\d+(?:-\d+)?(?:\s*\|\s*(?:text|table))?\]")

def ensure_citations(answer: str) -> str:
    # Appending a neutral verification note if no bracket-style citations are present
    text = (answer or "").strip()
    if not _CITE_PATTERN.search(text):
        return text + "\n\n[Note: No in-text citations detected, verify against the provided context.]"
    return text


E5_MODEL_ID = os.environ.get("E5_MODEL_ID", "intfloat/e5-base-v2")

_E5_MODEL = None  # lazy cache

def _load_e5():
    global _E5_MODEL
    if _E5_MODEL is None:    
        _E5_MODEL = SentenceTransformer(E5_MODEL_ID, device="cpu")
    return _E5_MODEL

def embed_query_e5(text: str) -> np.ndarray:
    model = _load_e5()
    vec = model.encode([f"query: {text}"], normalize_embeddings=True)  # shape (1, D), L2-normalized
    return vec.astype(np.float32)

# loading artifacts
def load_artifacts(
    faiss_path: Path,
    meta_path: Path,
    emb_path: Path,
    bm25_path: Path,
) -> Dict[str, Any]:
    # load FAISS
    
    index = faiss.read_index(str(faiss_path))

    # load metadata (must include columns used by retriever and context)
    meta = pd.read_parquet(meta_path)
    emb = np.load(str(emb_path)).astype(np.float32)
    # optional: load BM25 corpus/tokenizer if you saved it
    bm25 = None
    with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)

    # return a simple bag of objects for the retriever
    return {"faiss": index, "meta": meta, "emb": emb, "bm25": bm25, "embed_fn": embed_query_e5}

""" few functions used by search hybrid mmr"""
def _simple_tokens(t: str):
    # Same tokenizer you used to build BM25
    return re.findall(r"[A-Za-z0-9_]+", (t or "").lower())

def _minmax_norm(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xmax <= xmin:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - xmin) / (xmax - xmin)).astype(np.float32)

_NUM_PAT = re.compile(r"\d")
def _looks_numeric_query(q: str) -> bool:
    # When the question contains digits, push tables slightly
    return bool(_NUM_PAT.search(q))

def mmr_select(q_vec: np.ndarray, idx_pool: np.ndarray, vecs_pool: np.ndarray,
               k: int = 5, lambda_mult: float = 0.7):
    # Greedy MMR using cosine (dot product; vectors are normalized)
    q = q_vec.reshape(1, -1)
    picked = []
    remaining = list(range(len(idx_pool)))

    rel = (q @ vecs_pool.T).ravel()
    while remaining and len(picked) < k:
        if not picked:
            best = int(np.argmax(rel[remaining]))
            first = remaining.pop(best)
            picked.append(idx_pool[first])
            continue
        # Max similarity to any already-picked
        picked_mask = np.array([i for i, _ in enumerate(idx_pool) if idx_pool[i] in picked])
        if picked_mask.size == 0:
            picked_sims = np.zeros(len(remaining), dtype=np.float32)
        else:
            P = vecs_pool[picked_mask]
            sims = vecs_pool[remaining] @ P.T
            picked_sims = sims.max(axis=1)
        mmr = lambda_mult * rel[remaining] - (1.0 - lambda_mult) * picked_sims
        best = int(np.argmax(mmr))
        nxt = remaining.pop(best)
        picked.append(idx_pool[nxt])
    return picked


# -----------------------------
# 7) Hybrid + MMR retriever
# -----------------------------
# This is a thin wrapper that calls YOUR existing hybrid scoring + MMR logic.
# Paste your function body where indicated. Keep the signature the same.
def search_hybrid_mmr(
    query: str,
    k: int = 5,
    pool_dense: int = 40,
    pool_bm25: int = 80,
    w_dense: float = 0.6,
    w_bm25: float = 0.4,
    boost_tables: float = 0.10,
    lambda_mult: float = 0.7,
    artifacts: dict | None = None,
) -> pd.DataFrame:
    assert artifacts is not None, "artifacts dict is required"

    index = artifacts["faiss"]
    df    = artifacts["meta"]
    emb   = artifacts["emb"]
    bm25  = artifacts.get("bm25", None)
    embed_fn = artifacts.get("embed_fn", embed_query_e5)  # <= guarantees availability

    # 1) Encode query (same E5 family as your index)
    qv = embed_fn(query)                       # (1, D), normalized

    # 2) Dense pool (FAISS)
    D, I = index.search(qv, pool_dense)       # D: scores, I: row indices
    dense_idx, dense_score = I[0], D[0]

    # 3) BM25 pool (optional)
    if bm25 is not None:
        q_toks = _simple_tokens(query)
        bm25_scores_all = bm25.get_scores(q_toks)                 # (N,)
        bm25_idx   = np.argsort(bm25_scores_all)[::-1][:pool_bm25]
        bm25_score = bm25_scores_all[bm25_idx]
    else:
        bm25_scores_all = np.zeros(len(df), dtype=np.float32)
        bm25_idx   = np.array([], dtype=np.int64)
        bm25_score = np.array([], dtype=np.float32)

    # 4) Merge candidates
    ddf  = pd.DataFrame({"row_id": dense_idx, "score_dense": dense_score})
    bdf  = pd.DataFrame({"row_id": bm25_idx,  "score_bm25": bm25_score})
    cand = pd.merge(ddf, bdf, on="row_id", how="outer").fillna(0.0)

    # 5) Normalize + optional table boost
    cand["dense_norm"] = _minmax_norm(cand["score_dense"].to_numpy())
    cand["bm25_norm"]  = _minmax_norm(cand["score_bm25"].to_numpy())
    if bm25 is not None and _looks_numeric_query(query):
        is_table = df.iloc[cand["row_id"]]["content_type"].eq("table").to_numpy(dtype=bool)
        cand.loc[is_table, "bm25_norm"] = np.minimum(1.0, cand.loc[is_table, "bm25_norm"] + float(boost_tables))

    # 6) Hybrid blend
    cand["score_hybrid"] = float(w_dense) * cand["dense_norm"] + float(w_bm25) * cand["bm25_norm"]

    # 7) MMR rerank on dense vectors
    pool_idx  = cand.sort_values("score_hybrid", ascending=False)["row_id"].to_numpy()
    pool_vecs = emb[pool_idx]  # (P, D)
    picked    = mmr_select(qv, pool_idx, pool_vecs, k=int(k), lambda_mult=float(lambda_mult))

    # 8) Build final DataFrame + transparent scores
    out = df.iloc[picked].copy()
    out["score_dense"] = (qv @ emb[picked].T).ravel()
    out["score_bm25"]  = bm25_scores_all[picked] if bm25 is not None else 0.0
    out["dense_norm"]  = _minmax_norm(out["score_dense"].to_numpy())
    out["bm25_norm"]   = _minmax_norm(np.asarray(out["score_bm25"]).astype(float))
    out["score_hybrid"] = float(w_dense) * out["dense_norm"] + float(w_bm25) * out["bm25_norm"]

    # Add citation string you already use
    out["citation"] = out.apply(cite_str, axis=1)

    cols = [
        "score_hybrid", "dense_norm", "bm25_norm", "score_dense", "score_bm25",
        "citation", "heading", "content_type", "text",
        "doc_label", "page_start", "page_end", "chunk_id"
    ]
    cols = [c for c in cols if c in out.columns]
    return out[cols]

# # generate answer function
# def generate_answer_hf(prompt: str, model_id: str = "Qwen/Qwen2.5-3B-Instruct", max_new_tokens: int = 400) -> str:
#     # Loads the model/pipeline and generates deterministically (do_sample=False)
#     src = _model_source()
#     try:
#         from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
#         tok = AutoTokenizer.from_pretrained(
#             src,
#             local_files_only=bool('USE_LOCAL' in globals() and USE_LOCAL))

#         mdl = AutoModelForCausalLM.from_pretrained(
#             src,
#             local_files_only=bool('USE_LOCAL' in globals() and USE_LOCAL),
#             device_map="auto",
#             torch_dtype="auto"
#         )
#         pipe = pipeline("text-generation", model=mdl, tokenizer=tok)
#         out = pipe(
#             prompt,
#             max_new_tokens=max_new_tokens,
#             do_sample=False,
#             temperature=0.0,
#             pad_token_id=tok.eos_token_id,
#         )
#         # Extract everything after the explicit tag to keep outputs clean
#         return out[0]["generated_text"].split("Answer (with citations):")[-1].strip()
#     except Exception as e:
#         return f"[LLM error] {e}"




HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "").strip()
HF_GEN_MODEL = os.environ.get("HF_GEN_MODEL", "Qwen/Qwen2.5-3B-Instruct").strip()

def _hf_api_generate(prompt: str, model_id: str, max_new_tokens: int = 250, temperature: float = 0.0, timeout: int = 60) -> str:
    if not HF_API_TOKEN:
        return "[LLM error] HF_API_TOKEN is not set. Add it in Space settings."
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "return_full_text": True,
            "do_sample": temperature > 0
        },
        "options": {"wait_for_model": True}
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code == 503:  # warming
            time.sleep(2); resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"]
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        return f"[LLM error] Unexpected HF API response: {str(data)[:200]}"
    except Exception as e:
        return f"[LLM error] {e}"

def generate_answer_hf(prompt: str, model_id: str = None, max_new_tokens: int = 400) -> str:
    mid = model_id or HF_GEN_MODEL
    raw = _hf_api_generate(prompt, model_id=mid, max_new_tokens=max_new_tokens, temperature=0.0)
    if raw.startswith("[LLM error]"):
        return raw
    return raw.split("Answer (with citations):")[-1].strip()
