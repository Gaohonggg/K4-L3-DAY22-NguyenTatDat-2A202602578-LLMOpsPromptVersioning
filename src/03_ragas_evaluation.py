"""Step 3: evaluate 50 QA pairs against each pinned Hub prompt with RAGAS.

Run from the repository root:
    .venv/bin/python src/03_ragas_evaluation.py --limit 2
    .venv/bin/python src/03_ragas_evaluation.py

Successful generations and evaluation batches are checkpointed under logs/.
Rerunning the same command resumes the same configuration automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import config  # Configure tracing before importing LangChain/RAGAS.

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tracers.langchain import wait_for_all_tracers
from langsmith import Client
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig

from qa_pairs import QA_PAIRS
from utils.data_loader import build_vectorstore, load_knowledge_base, split_text
from utils.llm_factory import get_embeddings, get_llm
from utils.prompt_registry import PROMPT_NAMES
from utils.ragas_support import (
    METRIC_NAMES,
    build_ragas_dataset,
    normalize_scores,
    summarize_scores,
    validate_contexts,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_BATCH_SIZE = 5


def load_pinned_prompts(client: Client) -> tuple[dict[str, ChatPromptTemplate], dict[str, Any]]:
    """Pull exactly the immutable prompt commits recorded by step 2."""
    path = ROOT / "logs" / "02_prompt_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("source") != "hub":
        raise ValueError("Step 2 manifest must identify Hub prompts")
    pinned = manifest["versions"]
    prompts = {}
    for tag, name in PROMPT_NAMES.items():
        entry = pinned[tag]
        commit_hash = entry["commit_hash"]
        if entry["name"] != name or not re.fullmatch(r"[0-9a-f]{64}", commit_hash):
            raise ValueError("Invalid pinned prompt name or commit hash")
        identifier = f"{name}:{commit_hash}"
        if entry["identifier"] != identifier:
            raise ValueError("Pinned prompt identifier does not match its commit")
        prompt = client.pull_prompt(identifier, include_model=False, skip_cache=True)
        if not isinstance(prompt, ChatPromptTemplate):
            raise TypeError("Hub returned an unexpected prompt object")
        if set(prompt.input_variables) != {"context", "question"}:
            raise ValueError("Pinned prompt has unexpected input variables")
        if (prompt.metadata or {}).get("lc_hub_commit_hash") != commit_hash:
            raise ValueError("Hub returned a different commit")
        prompts[tag] = prompt
        print(f"PULL {tag} from Hub | identifier={identifier}", flush=True)
    return prompts, pinned


def open_checkpoint(settings: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Separate smoke/full configurations and reject mismatched checkpoint data."""
    fingerprint = hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode("utf-8")
    ).hexdigest()
    folder = ROOT / "logs" / "03_ragas" / fingerprint[:16]
    path = folder / "checkpoint.json"
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("settings") != settings:
            raise ValueError("Checkpoint settings mismatch")
        print(f"RESUME checkpoint={path}", flush=True)
    else:
        state = {
            "settings": settings,
            "contexts": [],
            "outputs": {"v1": [], "v2": []},
            "scores": {"v1": [], "v2": []},
        }
        write_json(path, state)
        print(f"NEW checkpoint={path}", flush=True)
    return path, state


def collect_contexts(
    questions: list[dict[str, str]], text: str, state: dict[str, Any], checkpoint: Path,
) -> None:
    """Retrieve once per question so V1 and V2 receive identical passages."""
    cached = state["contexts"]
    if len(cached) > len(questions):
        raise ValueError("Checkpoint contains too many retrieved samples")
    for index, contexts in enumerate(cached):
        validate_contexts(contexts)
    if len(cached) == len(questions):
        print(f"CACHE retrieval: {len(cached)} questions", flush=True)
        return

    chunks = split_text(text, chunk_size=500, chunk_overlap=50)
    if not chunks:
        raise ValueError("Knowledge base produced no chunks")
    retriever = build_vectorstore(chunks, get_embeddings()).as_retriever(search_kwargs={"k": 3})
    for index in range(len(cached), len(questions)):
        docs = retriever.invoke(questions[index]["question"])
        contexts = validate_contexts([doc.page_content for doc in docs])
        cached.append(contexts)
        write_json(checkpoint, state)
        print(f"RETRIEVE [{index + 1:02d}/{len(questions):02d}]", flush=True)


def collect_rag_outputs(
    questions: list[dict[str, str]], prompts: dict[str, ChatPromptTemplate],
    state: dict[str, Any], checkpoint: Path,
) -> None:
    """Generate both variants; keep references out of every generation prompt."""
    llm = get_llm(temperature=0)
    for tag, prompt in prompts.items():
        records = state["outputs"][tag]
        if len(records) > len(questions):
            raise ValueError("Checkpoint contains too many generated answers")
        for index, record in enumerate(records):
            if record["question"] != questions[index]["question"]:
                raise ValueError("Cached question mismatch")
            if record["reference"] != questions[index]["reference"]:
                raise ValueError("Cached reference mismatch")
            if record["contexts"] != state["contexts"][index]:
                raise ValueError("Cached retrieval mismatch")
        if records:
            build_ragas_dataset(records)
        print(f"CACHE {tag} generation: {len(records)}/{len(questions)}", flush=True)
        chain = (prompt | llm | StrOutputParser()).with_config(
            run_name=f"ragas-generate-{tag}",
            tags=["step3", f"prompt-{tag}"],
            metadata={"prompt_commit": prompt.metadata["lc_hub_commit_hash"]},
        )
        for index in range(len(records), len(questions)):
            qa = questions[index]
            contexts = state["contexts"][index]
            answer = chain.invoke({
                "context": "\n\n".join(contexts),
                "question": qa["question"],
            })
            record = {
                "question": qa["question"],
                "reference": qa["reference"],
                "answer": answer,
                "contexts": contexts,
            }
            build_ragas_dataset([record])
            records.append(record)
            write_json(checkpoint, state)
            print(f"GENERATE {tag} [{index + 1:02d}/{len(questions):02d}] {qa['question']}", flush=True)


def run_ragas_eval(
    tag: str, state: dict[str, Any], checkpoint: Path, max_workers: int,
) -> dict[str, float]:
    """Evaluate resumable batches; each saved sample must have four finite scores."""
    records = state["outputs"][tag]
    saved = state["scores"][tag]
    if len(saved) > len(records):
        raise ValueError("Checkpoint contains too many evaluated samples")
    if saved:
        normalize_scores(saved, len(saved))
    print(f"CACHE {tag} evaluation: {len(saved)}/{len(records)}", flush=True)
    llm = get_llm(temperature=0)
    embeddings = get_embeddings()
    run_config = RunConfig(timeout=180, max_retries=3, max_workers=max_workers, seed=42)

    while len(saved) < len(records):
        start = len(saved)
        batch = records[start:start + EVALUATION_BATCH_SIZE]
        print(f"EVALUATE {tag} samples {start + 1}-{start + len(batch)}/{len(records)}", flush=True)
        # Fresh instances avoid leaking evaluator state between batches/variants.
        result = evaluate(
            build_ragas_dataset(batch),
            metrics=[
                Faithfulness(),
                ResponseRelevancy(),
                LLMContextRecall(),
                LLMContextPrecisionWithReference(name="context_precision"),
            ],
            llm=llm,
            embeddings=embeddings,
            run_config=run_config,
            raise_exceptions=True,
            show_progress=True,
            experiment_name=f"day22-{tag}",
        )
        scores = normalize_scores(result.scores, len(batch))
        saved.extend(scores)
        write_json(checkpoint, state)
    return summarize_scores(saved, len(records))


def make_report(state: dict[str, Any], smoke: bool) -> dict[str, Any]:
    """Build a strict JSON report including aggregate and per-sample evidence."""
    count = state["settings"]["sample_count_per_version"]
    scores = {tag: summarize_scores(state["scores"][tag], count) for tag in PROMPT_NAMES}
    best_faithfulness = max(values["faithfulness"] for values in scores.values())
    per_sample = {
        tag: [
            {**record, "scores": metrics}
            for record, metrics in zip(state["outputs"][tag], state["scores"][tag], strict=True)
        ]
        for tag in PROMPT_NAMES
    }
    return {
        "prompt_v1_scores": scores["v1"],
        "prompt_v2_scores": scores["v2"],
        "target_met": best_faithfulness >= 0.8,
        "bonus_faithfulness_met": all(values["faithfulness"] >= 0.9 for values in scores.values()),
        "evaluation_complete": True,
        "submission_ready": not smoke and count == 50 and best_faithfulness >= 0.8,
        "mode": "smoke" if smoke else "full",
        "sample_count_per_version": count,
        "evaluated_samples_total": count * 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": state["settings"],
        "per_sample": per_sample,
    }


def print_comparison(report: dict[str, Any]) -> None:
    """Print an untruncated comparison suitable for a terminal screenshot."""
    print("\n" + "=" * 76)
    print(f"RAGAS RESULTS | mode={report['mode']} | samples/version={report['sample_count_per_version']}")
    print(f"{'Metric':30s} {'V1':>10s} {'V2':>10s} {'Winner':>10s}")
    print("-" * 76)
    for metric in METRIC_NAMES:
        v1 = report["prompt_v1_scores"][metric]
        v2 = report["prompt_v2_scores"][metric]
        winner = "Tie" if abs(v1 - v2) < 1e-9 else ("V1" if v1 > v2 else "V2")
        print(f"{metric:30s} {v1:10.4f} {v2:10.4f} {winner:>10s}")
    print(f"Target faithfulness >=0.8: {report['target_met']}")
    print(f"Bonus faithfulness >=0.9 for BOTH: {report['bonus_faithfulness_met']}")
    print(f"Submission ready: {report['submission_ready']}")
    print("=" * 76, flush=True)


def main(*, limit: int | None = None, max_workers: int = 2) -> None:
    """Run smoke or full evaluation and resume completed checkpoints automatically."""
    if limit is not None and not 1 <= limit <= len(QA_PAIRS):
        raise ValueError(f"limit must be between 1 and {len(QA_PAIRS)}")
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    if not config.validate():
        raise SystemExit(1)
    if config.PROVIDER != "openai":
        raise ValueError("This evaluation configuration requires PROVIDER=openai")
    if limit is None and len(QA_PAIRS) != 50:
        raise ValueError("The full lab requires exactly 50 QA pairs")

    stage = "pinned Hub prompt loading"
    checkpoint = None
    try:
        client = Client(api_key=config.LANGSMITH_API_KEY, api_url=os.environ["LANGCHAIN_ENDPOINT"])
        prompts, pinned = load_pinned_prompts(client)
        questions = QA_PAIRS if limit is None else QA_PAIRS[:limit]
        text = load_knowledge_base()
        settings = {
            "schema_version": 1,
            "sample_count_per_version": len(questions),
            "mode": "full" if limit is None else "smoke",
            "prompts": pinned,
            "provider": config.PROVIDER,
            "generation_model": config.OPENAI_MODEL,
            "evaluator_model": config.OPENAI_MODEL,
            "embedding_model": config.OPENAI_EMBEDDING_MODEL,
            "openai_base_url": config.OPENAI_BASE_URL,
            "temperature": 0,
            "chunk_size": 500,
            "chunk_overlap": 50,
            "retrieval_k": 3,
            "knowledge_base_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "qa_sha256": hashlib.sha256(json.dumps(questions, sort_keys=True).encode("utf-8")).hexdigest(),
            "library_versions": {
                name: version(name)
                for name in ("ragas", "langchain-core", "langchain-openai", "langchain-community", "faiss-cpu")
            },
            "metrics": list(METRIC_NAMES),
            "metric_implementation": "ragas-legacy-single-turn",
            "evaluation_batch_size": EVALUATION_BATCH_SIZE,
            "evaluator_seed": 42,
        }
        stage = "checkpoint/retrieval"
        checkpoint, state = open_checkpoint(settings)
        collect_contexts(questions, text, state, checkpoint)
        stage = "answer generation"
        collect_rag_outputs(questions, prompts, state, checkpoint)
        for tag in PROMPT_NAMES:
            stage = f"RAGAS evaluation {tag}"
            run_ragas_eval(tag, state, checkpoint, max_workers)

        stage = "report export"
        report = make_report(state, smoke=limit is not None)
        if limit is None:
            paths = [ROOT / "data" / "ragas_report.json", ROOT / "evidence" / "03_ragas_report.json"]
        else:
            paths = [checkpoint.parent / "smoke_report.json"]
        for report_path in paths:
            write_json(report_path, report)
            print(f"SAVED {report_path}", flush=True)
        print_comparison(report)
        if limit is None and not report["target_met"]:
            print("Evaluation complete, but quality target not met. Keep the report for diagnosis.")
            raise SystemExit(1)
    except Exception as error:
        print(f"FAIL {stage} | error={type(error).__name__}", flush=True)
        if checkpoint is not None:
            print(f"Completed work preserved: {checkpoint}. Rerun the same command after resolving the error.")
        raise SystemExit(1) from None
    finally:
        wait_for_all_tracers()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, choices=range(1, len(QA_PAIRS) + 1))
    parser.add_argument("--max-workers", type=int, default=2, help="Concurrent RAGAS metric tasks (default: 2)")
    args = parser.parse_args()
    main(limit=args.limit, max_workers=args.max_workers)
