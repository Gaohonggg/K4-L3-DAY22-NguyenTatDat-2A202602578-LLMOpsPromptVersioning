"""Check local dependencies; opt in to a small OpenAI/LangSmith API smoke check.

Run from the repository root:
    .venv/bin/python src/00_check_setup.py
    .venv/bin/python src/00_check_setup.py --online

The online check makes one embedding request and one LLM request. Its trace
is tagged setup and is separate from the required step 1/2 traces.
"""

import argparse
import importlib
import logging
import os
import sys
from importlib.metadata import version

import config  # Configure tracing before importing LangChain or LangSmith.


LOGGER = logging.getLogger(__name__)
DEPENDENCIES = {
    "langchain": "langchain",
    "langchain_core": "langchain-core",
    "langchain_community": "langchain-community",
    "langchain_text_splitters": "langchain-text-splitters",
    "langchain_openai": "langchain-openai",
    "langsmith": "langsmith",
    "faiss": "faiss-cpu",
    "ragas": "ragas",
    "guardrails": "guardrails-ai",
    "numpy": "numpy",
}


def check_dependencies() -> bool:
    """Import required modules and check the APIs used by the lab scaffold."""
    passed = True
    for module_name, distribution_name in DEPENDENCIES.items():
        try:
            importlib.import_module(module_name)
            LOGGER.info("PASS dependency %s=%s", distribution_name, version(distribution_name))
        except ModuleNotFoundError as error:
            LOGGER.error("FAIL import %s (missing module: %s)", module_name, error.name)
            passed = False
        except Exception as error:
            LOGGER.error("FAIL import %s (%s)", module_name, type(error).__name__)
            passed = False

    if not passed:
        return False

    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
        from guardrails.validators import FailResult, PassResult, Validator, register_validator
        from langchain_community.vectorstores import FAISS
        from langsmith import Client, traceable

        # Imports above check the scaffold's API surface without making requests.
        LOGGER.info("PASS lab API imports")
    except Exception as error:
        LOGGER.error("FAIL lab API imports (%s)", type(error).__name__)
        return False
    return True


def check_online() -> bool:
    """Check authentication, embeddings, generation, and trace submission."""
    from langsmith import Client, traceable
    from utils.llm_factory import get_embeddings, get_llm

    stage = "LangSmith authentication"
    try:
        client = Client(
            api_key=config.LANGSMITH_API_KEY,
            api_url=os.environ["LANGCHAIN_ENDPOINT"],
        )
        # Consume the iterator so a request is made even in an empty workspace.
        list(client.list_projects(limit=1))
        LOGGER.info("PASS %s", stage)

        stage = "OpenAI embeddings"
        vector = get_embeddings().embed_query("Environment setup check")
        if not vector:
            raise ValueError("Empty embedding")
        LOGGER.info("PASS %s (dimensions=%d)", stage, len(vector))

        stage = "OpenAI generation"

        @traceable(name="setup-check", tags=["setup"], client=client)
        def generate_check() -> str:
            """Make one short, traced request without sensitive input data."""
            response = get_llm(temperature=0).invoke([
                ("system", "Respond with only SETUP_OK."),
                ("human", "Check that generation works."),
            ])
            if not response.content:
                raise ValueError("Empty LLM response")
            return str(response.content)

        generate_check()
        LOGGER.info("PASS %s", stage)
        stage = "LangSmith trace submission"
        client.flush()
        LOGGER.info("Trace queue flushed; verify setup-check in the LangSmith dashboard")
    except Exception as error:
        # Avoid logging provider exception bodies, which may contain credentials.
        LOGGER.error("FAIL %s (%s); check keys, model access, quota and network", stage, type(error).__name__)
        return False
    return True


def main() -> int:
    """Return a nonzero exit status when a setup check fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online", action="store_true", help="Make small, billable API requests")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout)
    LOGGER.info("Python %s | executable=%s", sys.version.split()[0], sys.executable)
    if sys.version_info < (3, 10):
        LOGGER.error("Python >=3.10 is required")
        return 1
    if not check_dependencies() or not config.validate():
        return 1
    if args.online:
        if config.PROVIDER != "openai":
            LOGGER.error("This setup check requires PROVIDER=openai")
            return 1
        if not check_online():
            return 1
    LOGGER.info("SETUP CHECK PASSED (%s)", "online" if args.online else "local")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
