"""Step 2: publish/pull two Hub prompts and run deterministic A/B queries.

Run from the repository root:
    .venv/bin/python src/02_prompt_hub_ab_routing.py --limit 5
    .venv/bin/python src/02_prompt_hub_ab_routing.py

The full run processes 50 questions. Hub failures stop the run.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, TypedDict
from uuid import uuid4

import config  # Configure tracing before importing LangChain/LangSmith.

from langchain_community.vectorstores import FAISS
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.tracers.langchain import wait_for_all_tracers
from langsmith import Client, traceable

from qa_pairs import SAMPLE_QUESTIONS
from utils.data_loader import build_vectorstore, load_knowledge_base, split_text
from utils.llm_factory import get_embeddings, get_llm
from utils.prompt_registry import (
    PROMPT_NAMES,
    PROMPT_V1_NAME,
    get_prompt_version,
    pull_prompts_from_hub,
    push_prompts_to_hub,
)


class ABResult(TypedDict):
    """Serializable trace output containing the generated answer and its sources."""

    question: str
    answer: str
    contexts: list[str]
    version: str
    request_id: str


def trace_query_inputs(inputs: dict[str, Any]) -> dict[str, str]:
    """Keep query identifiers while excluding model/retriever objects."""
    return {key: inputs[key] for key in ("question", "version", "request_id")}


@traceable(name="ab-rag-query", tags=["ab-test", "step2"], process_inputs=trace_query_inputs)
def ask_ab(
    retriever: BaseRetriever,
    llm: BaseChatModel,
    prompt: ChatPromptTemplate,
    question: str,
    version: str,
    request_id: str,
) -> ABResult:
    """Retrieve once and generate with the selected Hub prompt."""
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    if not contexts:
        raise ValueError("Retrieval returned no context")
    chain = (prompt | llm | StrOutputParser()).with_config(run_name=f"rag-{version}")
    answer = chain.invoke({"context": "\n\n".join(contexts), "question": question})
    if not answer.strip():
        raise ValueError("The LLM returned an empty answer")
    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "version": version,
        "request_id": request_id,
    }


def setup_vectorstore() -> FAISS:
    """Build the same retrieval baseline as step 1: 500/50 chunks, top-k 3."""
    text = load_knowledge_base()
    if not text.strip():
        raise ValueError("The knowledge base is empty")
    chunks = split_text(text, chunk_size=500, chunk_overlap=50)
    if not chunks:
        raise ValueError("Chunking produced no passages")
    print(f"Knowledge base: {len(chunks)} chunks | chunk_size=500 | overlap=50", flush=True)
    return build_vectorstore(chunks, get_embeddings())


def save_prompt_manifest(prompts: dict[str, ChatPromptTemplate], batch_id: str) -> None:
    """Persist exact Hub identifiers for later evaluation without copying templates."""
    versions = {}
    for version, name in PROMPT_NAMES.items():
        commit_hash = prompts[name].metadata["lc_hub_commit_hash"]
        versions[version] = {
            "name": name,
            "commit_hash": commit_hash,
            "identifier": f"{name}:{commit_hash}",
        }
    path = Path(__file__).resolve().parents[1] / "logs" / "02_prompt_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"batch_id": batch_id, "source": "hub", "versions": versions}, indent=2),
        encoding="utf-8",
    )
    print(f"Prompt manifest saved: {path}", flush=True)


def main(*, limit: int | None = None) -> None:
    """Run the Hub-backed A/B batch; callable by run_all.py without CLI parsing."""
    if limit is not None and not 1 <= limit <= len(SAMPLE_QUESTIONS):
        raise ValueError(f"limit must be between 1 and {len(SAMPLE_QUESTIONS)}")
    if not config.validate():
        raise SystemExit(1)

    questions = SAMPLE_QUESTIONS if limit is None else SAMPLE_QUESTIONS[:limit]
    batch_id = str(uuid4())
    print(f"Step 2 | project={config.LANGSMITH_PROJECT} | batch_id={batch_id}", flush=True)
    print(f"Mode={'full' if limit is None else 'smoke'} | questions={len(questions)}", flush=True)
    client = Client(
        api_key=config.LANGSMITH_API_KEY,
        api_url=os.environ["LANGCHAIN_ENDPOINT"],
    )
    counts = {"v1": 0, "v2": 0}
    succeeded = 0
    failed: list[str] = []
    stage = "Hub push/pull"
    try:
        push_prompts_to_hub(client)
        prompts = pull_prompts_from_hub(client)
        save_prompt_manifest(prompts, batch_id)

        stage = "FAISS/LLM setup"
        retriever = setup_vectorstore().as_retriever(search_kwargs={"k": 3})
        llm = get_llm(temperature=0)
        stage = "query execution"
        for index, question in enumerate(questions):
            request_id = f"req-{index:04d}"
            name = get_prompt_version(request_id)
            version = "v1" if name == PROMPT_V1_NAME else "v2"
            prompt = prompts[name]
            counts[version] += 1
            print(
                f"\n[{index + 1:02d}/{len(questions):02d}] [prompt-{version}] "
                f"request_id={request_id} | Q: {question}",
                flush=True,
            )
            try:
                result = ask_ab(
                    retriever, llm, prompt, question, version, request_id,
                    langsmith_extra={
                        "client": client,
                        "tags": [f"prompt-{version}"],
                        "metadata": {
                            "batch_id": batch_id,
                            "mode": "full" if limit is None else "smoke",
                            "request_id": request_id,
                            "prompt_version": version,
                            "prompt_name": name,
                            "prompt_commit": prompt.metadata["lc_hub_commit_hash"],
                            "prompt_source": "hub",
                        },
                    },
                )
                succeeded += 1
                print(f"A: {result['answer']}", flush=True)
            except Exception as error:
                failed.append(request_id)
                print(f"FAIL request_id={request_id} | error={type(error).__name__}", flush=True)
    except Exception as error:
        print(f"FAIL {stage} | error={type(error).__name__}", flush=True)
        raise SystemExit(1) from None
    finally:
        try:
            wait_for_all_tracers()
            client.flush()
        except Exception as error:
            print(f"FAIL trace flush | error={type(error).__name__}", flush=True)
            raise SystemExit(1) from None

    print(f"\nRouting: V1={counts['v1']} | V2={counts['v2']} | total={len(questions)}")
    print(f"Summary: succeeded={succeeded} | failed={len(failed)} | total={len(questions)}")
    print(f"Trace queue flushed. Verify ab-rag-query traces for batch_id={batch_id} in LangSmith.")
    if failed or (limit is None and not all(counts.values())):
        print(f"Batch incomplete: failed_request_ids={failed}; both versions must receive queries.")
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, choices=range(1, len(SAMPLE_QUESTIONS) + 1),
        help="Run the first N questions for a smoke check; omit for the full lab",
    )
    main(limit=parser.parse_args().limit)
