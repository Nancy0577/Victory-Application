"""
demo_llm_rag_copilot.py

A minimal RAG + LLM copilot demo:
1) build/reuse a FAISS index over local docs (course notes/policies/snippets)
2) retrieve top-k evidence for a user query
3) call LLM API and request STRICT JSON output
4) parse JSON, validate schema, and do follow-up processing:
   - attach evidence
   - generate a markdown report
   - save output to disk
"""

import os
import json
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from pydantic import BaseModel, Field, ValidationError

# --- If you use OpenAI ---
from openai import OpenAI

# If you want to swap to Anthropic (Claude), you'd implement a similar client call.
# from anthropic import Anthropic

# -----------------------------
# 1) Data structures
# -----------------------------

@dataclass
class DocChunk:
    doc_id: str
    text: str

class PlanJSON(BaseModel):
    """Expected structured JSON returned by the LLM."""
    title: str = Field(..., description="Short title of the solution plan")
    summary: str = Field(..., description="1-2 paragraph explanation")
    steps: List[str] = Field(..., description="Ordered implementation steps")
    files: List[Dict[str, str]] = Field(..., description="List of files to create/modify with purpose")
    tests: List[str] = Field(..., description="Proposed tests")
    risks: List[str] = Field(..., description="Risks & mitigations")
    references: List[Dict[str, str]] = Field(..., description="Evidence references injected by the system")

# -----------------------------
# 2) Embedding + FAISS index
# -----------------------------

def _hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

def chunk_text(text: str, max_chars: int = 800) -> List[str]:
    text = text.strip()
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def embed_texts_openai(client: OpenAI, texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """Return embeddings as float32 numpy array [n, d]."""
    resp = client.embeddings.create(model=model, input=texts)
    vectors = np.array([item.embedding for item in resp.data], dtype=np.float32)
    return vectors

class SimpleRAG:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # cosine-like if vectors are normalized
        self.chunks: List[DocChunk] = []

    def add(self, vectors: np.ndarray, chunks: List[DocChunk]):
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vec: np.ndarray, top_k: int = 4) -> List[DocChunk]:
        faiss.normalize_L2(query_vec)
        scores, idx = self.index.search(query_vec, top_k)
        results = []
        for i in idx[0]:
            if i >= 0:
                results.append(self.chunks[i])
        return results

def build_demo_index(client: OpenAI, docs: Dict[str, str]) -> SimpleRAG:
    """
    docs: {doc_name: doc_text}
    """
    all_chunks: List[DocChunk] = []
    chunk_texts: List[str] = []

    for name, text in docs.items():
        for ch in chunk_text(text):
            cid = f"{name}:{_hash_text(ch)}"
            all_chunks.append(DocChunk(doc_id=cid, text=ch))
            chunk_texts.append(ch)

    vecs = embed_texts_openai(client, chunk_texts)
    rag = SimpleRAG(dim=vecs.shape[1])
    rag.add(vecs, all_chunks)
    return rag

# -----------------------------
# 3) LLM call (JSON output) + parse
# -----------------------------

SYSTEM_PROMPT = """You are an IT & AI teaching copilot. 
Return STRICT JSON only (no markdown, no commentary).
The JSON must follow the schema exactly:
{
  "title": "...",
  "summary": "...",
  "steps": ["..."],
  "files": [{"path": "...", "purpose": "..."}],
  "tests": ["..."],
  "risks": ["..."],
  "references": [{"doc_id": "...", "quote": "..."}]
}
Do not include extra keys.
"""

def call_llm_json_openai(
    client: OpenAI,
    user_request: str,
    evidence: List[DocChunk],
    model: str = "gpt-4.1-mini"
) -> PlanJSON:
    """
    Calls LLM with evidence and requests strict JSON.
    Parses and validates with Pydantic.
    """
    evidence_block = "\n\n".join(
        [f"[{i+1}] {c.doc_id}\n{c.text}" for i, c in enumerate(evidence)]
    )

    user_prompt = f"""Task:
{user_request}

Evidence (cite ONLY from this evidence in references):
{evidence_block}

Instructions:
- Produce a practical plan for an IT/AI student project.
- Use the evidence to justify key steps.
- 'references' must contain doc_id and short quotes copied from evidence.
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},  # forces JSON object
        temperature=0.2,
    )

    raw = resp.choices[0].message.content
    try:
        parsed = json.loads(raw)
        plan = PlanJSON(**parsed)  # schema validation
        return plan
    except (json.JSONDecodeError, ValidationError) as e:
        raise RuntimeError(f"LLM returned invalid JSON: {e}\nRAW:\n{raw}")

# -----------------------------
# 4) Follow-up processing
# -----------------------------

def generate_markdown_report(plan: PlanJSON) -> str:
    lines = []
    lines.append(f"# {plan.title}\n")
    lines.append(plan.summary.strip() + "\n")
    lines.append("## Implementation Steps")
    for i, s in enumerate(plan.steps, 1):
        lines.append(f"{i}. {s}")
    lines.append("\n## Files")
    for f in plan.files:
        lines.append(f"- `{f['path']}` — {f['purpose']}")
    lines.append("\n## Tests")
    for t in plan.tests:
        lines.append(f"- {t}")
    lines.append("\n## Risks & Mitigations")
    for r in plan.risks:
        lines.append(f"- {r}")
    lines.append("\n## Evidence")
    for ref in plan.references:
        q = ref["quote"].replace("\n", " ").strip()
        lines.append(f"- **{ref['doc_id']}**: “{q}”")
    return "\n".join(lines) + "\n"

def save_output(plan: PlanJSON, out_dir: str = "./demo_output"):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "plan.json"), "w", encoding="utf-8") as f:
        json.dump(plan.model_dump(), f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(generate_markdown_report(plan))

# -----------------------------
# 5) End-to-end demo run
# -----------------------------

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment")

    client = OpenAI(api_key=api_key)

    # Example "course-site-like" docs: replace with your real unit guide snippets/policies.
    docs = {
        "VU_IT_Assessment_Policy": "All submitted work must acknowledge GenAI use. Evidence-based reasoning is required ...",
        "Unit_Software_Engineering": "This unit covers requirements, design, testing, CI workflows, and team-based delivery ...",
        "Unit_Cloud_App_Dev": "Students build cloud-native services, REST APIs, deployment pipelines, and monitoring ...",
        "Capstone_Guidelines": "Projects must demonstrate industry relevance, reproducibility, and clear evaluation metrics ...",
    }

    rag = build_demo_index(client, docs)

    user_request = (
        "Design a capstone project showing an industry-relevant LLM application for IT students. "
        "It must include LLM API calls, RAG, and evaluation, and propose a 6-week plan."
    )

    qvec = embed_texts_openai(client, [user_request])
    evidence = rag.search(qvec, top_k=4)

    plan = call_llm_json_openai(client, user_request, evidence, model="gpt-4.1-mini")
    save_output(plan)

    print("Demo completed. Outputs saved to ./demo_output (plan.json, report.md)")

if __name__ == "__main__":
    main()
