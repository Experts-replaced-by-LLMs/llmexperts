import pytest
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

from src.llmexperts.model import LLMClient

load_dotenv()

def test_llm_client():
    model = "unknown-model"
    with pytest.raises(Exception):
        llm = LLMClient(model, 10)


def test_llm_client_dry_run():
    """
    model = ["gpt-4o-2024-08-06", "claude-3-5-sonnet-20241022", "gemini-1.5-pro-002"]
    probabilities = [True, False]
    """

    models = ["gpt-4o-2024-08-06", "claude-3-5-sonnet-20241022", "gemini-1.5-pro-002"]
    max_tokens = 10
    human_message = "Tell me a story."

    prompt = [
        SystemMessage(content="This is a prompt for testing."),
        HumanMessage(content=human_message)
    ]
    for model in models:
        for probabilities in [True, False]:
            llm = LLMClient(model, max_tokens=max_tokens, probabilities=probabilities)
            res = llm.invoke(prompt, dry_run=True, dry_run_res="test")
            assert res.content == "test"
            res = llm.invoke(prompt, dry_run="Test tag")
            assert res.content == f"Test tag {human_message}"
            res = llm.invoke(prompt, dry_run="Test tag", dry_run_res="test")
            assert res.content == f"test"

    batch_size = 5
    batch_prompt = [
        [
            SystemMessage(content="This is a prompt for testing."),
            HumanMessage(content=f"{i} {human_message}")
        ] for i in range(batch_size)
    ]
    for model in models:
        llm = LLMClient(model, max_tokens=max_tokens)
        res = llm.batch(batch_prompt, dry_run=True, dry_run_res="test")
        assert len(res) == batch_size
        for r in res:
            assert r.content == "test"

@pytest.mark.api
def test_llm_client_invoke():
    max_tokens = 20
    models = ["gpt-4o-2024-08-06", "claude-3-5-sonnet-20241022", "gemini-1.5-pro-002"]
    human_message = "Tell me a story."

    prompts = [
        SystemMessage(content="This is a prompt for API testing."),
        HumanMessage(content=human_message)
    ]
    for model in models:
        for probabilities in [True, False]:
            llm = LLMClient(model, max_tokens=max_tokens, probabilities=probabilities)
            res = llm.invoke(prompts, dry_run=False)
            if model.startswith("gpt"):
                assert model == res.response_metadata["model_name"]
            elif model.startswith("claude"):
                assert model == res.response_metadata["model"]
            elif model.startswith("gemini"):
                assert "prompt_feedback" in res.response_metadata
            usage_metadata = res.usage_metadata
            assert usage_metadata["input_tokens"] > 0
            assert usage_metadata["output_tokens"] > 0

    batch_size = 3
    batch_prompt = [
        [
            SystemMessage(content="This is a prompt for testing."),
            HumanMessage(content=f"{i} {human_message}")
        ] for i in range(batch_size)
    ]
    for model in models:
        llm = LLMClient(model, max_tokens=max_tokens)
        responses = llm.batch(batch_prompt, dry_run=False)
        assert len(responses) == batch_size
        for res in responses:
            if model.startswith("gpt"):
                assert model == res.response_metadata["model_name"]
            elif model.startswith("claude"):
                assert model == res.response_metadata["model"]
            elif model.startswith("gemini"):
                assert "prompt_feedback" in res.response_metadata
            usage_metadata = res.usage_metadata
            assert usage_metadata["input_tokens"] > 0
            assert usage_metadata["output_tokens"] > 0
