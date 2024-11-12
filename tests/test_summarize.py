import json
import os
import pytest
import shutil

from dotenv import load_dotenv

from src.llmexperts.summarize import summarize_file

load_dotenv()

def test_summarize_file_output(text_file_folder, capfd):

    prompt_template = os.path.join(os.path.dirname(__file__), "prompts-summarize.yaml")
    input_filenames = os.listdir(text_file_folder)
    issues_to_summarize = ["issue_1", "issue_2"]
    model = "gpt-4o-2024-08-06"
    output_folder = "./output"

    for filename in input_filenames:
        summarize_file(
            os.path.join(text_file_folder, filename),
            prompt_template,
            issues_to_summarize, output_folder, model, dry_run=True,
            save_summary=False, save_log=False,log_dir=None
        )
    assert len(list(os.listdir(output_folder))) == 0
    shutil.rmtree(output_folder)

    for filename in input_filenames:
        summarize_file(
            os.path.join(text_file_folder, filename),
            prompt_template,
            issues_to_summarize, output_folder, model, dry_run=True,
            save_summary=True, save_log=False, log_dir=None
        )
    assert len(list(os.listdir(output_folder))) == len(input_filenames)
    shutil.rmtree(output_folder)

    for filename in input_filenames:
        summarize_file(
            os.path.join(text_file_folder, filename),
            prompt_template,
            issues_to_summarize, output_folder, model, dry_run=True,
            save_summary=False, save_log=True, log_dir=None
        )
    assert len(list(os.listdir(output_folder))) == len(input_filenames)
    for filename in os.listdir(output_folder):
        assert filename.endswith(".json") == True
    shutil.rmtree(output_folder)

    for filename in input_filenames:
        summarize_file(
            os.path.join(text_file_folder, filename),
            prompt_template,
            issues_to_summarize, output_folder, model, dry_run=True,
            save_summary=True, save_log=True, log_dir=None
        )
    assert len(list(os.listdir(output_folder))) == len(input_filenames)*2
    shutil.rmtree(output_folder)

    log_dir = "./summary_logs"
    for filename in input_filenames:
        summarize_file(
            os.path.join(text_file_folder, filename),
            prompt_template,
            issues_to_summarize, output_folder, model, dry_run=True,
            save_summary=True, save_log=True, log_dir=log_dir
        )
    assert len(list(os.listdir(output_folder))) == len(input_filenames)
    assert len(list(os.listdir(log_dir))) == len(input_filenames)
    for filename in os.listdir(log_dir):
        assert filename.endswith(".json") == True

    for filename in input_filenames:
        summarize_file(
            os.path.join(text_file_folder, filename),
            prompt_template,
            issues_to_summarize, output_folder, model, dry_run=True,
            save_summary=True, save_log=True, log_dir=log_dir
        )
    out, err = capfd.readouterr()
    for filename in os.listdir(output_folder):
        assert f"Summary file {filename} already exists. Reusing the existing summary." in out
    assert len(list(os.listdir(output_folder))) == len(input_filenames)
    assert len(list(os.listdir(log_dir))) == len(input_filenames)
    for filename in os.listdir(log_dir):
        assert filename.endswith(".json") == True
    shutil.rmtree(output_folder)
    shutil.rmtree(log_dir)

def test_summarize_file_single_issue_dry_run(output_folder, text_file_folder):
    prompt_template = os.path.join(os.path.dirname(__file__), "prompts-summarize.yaml")
    input_filenames = os.listdir(text_file_folder)
    common_args = dict(
        prompt_template=prompt_template,
        output_dir=output_folder,
        try_no_chunk = False,
        chunk_size = 100000,
        overlap = 2500, min_size = 500, max_size = 1000,
        max_tokens_factor = 1.0,
        if_exists = 'reuse',
        save_summary = True,
        save_log = True,
        dry_run = True
    )

    issues_to_summarize = ["issue_1", "issue_2"]
    models = ["gpt-4o-2024-08-06", "claude-3-5-sonnet-20241022", "gemini-1.5-pro-002"]

    for filename in input_filenames:
        for model in models:
            for iss in issues_to_summarize:
                summarize_file(
                    os.path.join(text_file_folder, filename),
                    **{**common_args, "model": model, "issues_to_summarize": [iss]}
                )

    output_filenames = os.listdir(output_folder)

    assert len(output_filenames) == len(input_filenames)*len(models)*len(issues_to_summarize)*2
    for model in models:
        for filename in input_filenames:
            for iss in issues_to_summarize:
                assert os.path.isfile(os.path.join(output_folder, f"summary_standard__{model}__{iss}__{filename}")) == True
                assert os.path.isfile(os.path.join(output_folder, f"log_summary_standard__{model}__{iss}__{os.path.splitext(filename)[0]}.json")) == True

def test_summarize_file_multi_issue_dry_run(output_folder, text_file_folder):
    prompt_template = os.path.join(os.path.dirname(__file__), "prompts-summarize.yaml")
    input_filenames = os.listdir(text_file_folder)
    common_args = dict(
        prompt_template=prompt_template,
        output_dir=output_folder,
        try_no_chunk = False,
        chunk_size = 100000,
        overlap = 2500, min_size = 500, max_size = 1000,
        max_tokens_factor = 1.0,
        if_exists = 'reuse',
        save_summary = True,
        save_log = True,
        dry_run = True
    )

    issues_to_summarize = ["issue_1", "issue_2"]
    models = ["gpt-4o-2024-08-06", "claude-3-5-sonnet-20241022", "gemini-1.5-pro-002"]

    for filename in input_filenames:
        for model in models:
            summarize_file(
                os.path.join(text_file_folder, filename),
                **{**common_args, "model": model, "issues_to_summarize": issues_to_summarize}
            )

    output_filenames = os.listdir(output_folder)

    assert len(output_filenames) == len(input_filenames)*len(models)*2
    for model in models:
        for filename in input_filenames:
            assert os.path.isfile(os.path.join(output_folder, f"summary_standard__{model}__multi__{filename}")) == True
            assert os.path.isfile(os.path.join(output_folder, f"log_summary_standard__{model}__multi__{os.path.splitext(filename)[0]}.json")) == True

def test_summarize_file_multi_issue_dry_run_log_folder(output_folder, text_file_folder):
    prompt_template = os.path.join(os.path.dirname(__file__), "prompts-summarize.yaml")
    input_filenames = os.listdir(text_file_folder)
    common_args = dict(
        prompt_template=prompt_template,
        output_dir=output_folder,
        try_no_chunk=False,
        chunk_size=100000,
        overlap=2500, min_size=500, max_size=1000,
        max_tokens_factor=1.0,
        if_exists='reuse',
        save_summary=True,
        save_log=True,
        dry_run=True
    )

    issues_to_summarize = ["issue_1", "issue_2"]
    models = ["gpt-4o-2024-08-06", "claude-3-5-sonnet-20241022", "gemini-1.5-pro-002"]
    log_dir = os.path.join(output_folder, "summary_logs")

    for filename in input_filenames:
        for model in models:
            summarize_file(
                os.path.join(text_file_folder, filename),
                **{
                    **common_args,
                    "model": model, "issues_to_summarize": issues_to_summarize,"log_dir": log_dir
                }
            )

    output_filenames = os.listdir(output_folder)

    assert len(output_filenames) == len(input_filenames) * len(models) + 1
    assert len(os.listdir(log_dir)) == len(input_filenames) * len(models)

    for model in models:
        for filename in input_filenames:
            assert os.path.isfile(os.path.join(output_folder, f"summary_standard__{model}__multi__{filename}")) == True
            assert os.path.isfile(
                os.path.join(
                    output_folder, "summary_logs",
                    f"log_summary_standard__{model}__multi__{os.path.splitext(filename)[0]}.json")
            ) == True

@pytest.mark.api
def test_summarize_file_short_text():
    output_folder = os.path.join(os.path.dirname(__file__), "./test_summary_short_text_output")
    prompt_template = os.path.join(os.path.dirname(__file__), "prompts-summarize__real.yaml")
    text_dir = os.path.join(os.path.dirname(__file__), "./texts/short")
    input_filenames = os.listdir(text_dir)

    common_args = dict(
        prompt_template=prompt_template,
        issues_to_summarize=["taxation"],
        output_dir=output_folder,
        min_size=300, max_size=400,
        max_tokens_factor=2.0,
        if_exists='reuse',
        save_summary=True,
        save_log=True,
        dry_run=False
    )
    chunk_args = dict(
        try_no_chunk = False,
        chunk_size = 100000,
        overlap = 2500,
    )
    models = ["gpt-4o-2024-08-06", "claude-3-5-sonnet-20241022", "gemini-1.5-pro-002"]
    for filename in input_filenames:
        for model in models:
            summarize_file(
                os.path.join(text_dir, filename),
                **{**common_args, **chunk_args, "model": model}
            )
    for out_filename in os.listdir(output_folder):
        with open(os.path.join(output_folder, out_filename), "r", encoding="utf-8") as f:
            if out_filename.endswith(".txt"):
                content = f.read()
                assert len(content) > 0
            elif out_filename.endswith(".json"):
                _, model_used, issue_used, file_used = out_filename.split("__")
                content = json.loads(f.read())
                assert len(content["final_summary"]) > 0
                assert len(content["responses"]) == 1
                if model_used.startswith("gpt"):
                    assert model_used == content["responses"][0]["kwargs"]["response_metadata"]["model_name"]
                elif model_used.startswith("claude"):
                    assert model_used == content["responses"][0]["kwargs"]["response_metadata"]["model"]
                elif model_used.startswith("gemini"):
                    assert "prompt_feedback" in content["responses"][0]["kwargs"]["response_metadata"]
                assert content["responses"][0]["kwargs"]["usage_metadata"]["input_tokens"] > 0
                assert content["responses"][0]["kwargs"]["usage_metadata"]["output_tokens"] > 0

    shutil.rmtree(output_folder)


@pytest.mark.api
def test_summarize_file_long_texts(output_folder):
    prompt_template = os.path.join(os.path.dirname(__file__), "prompts-summarize__real.yaml")
    text_dir = os.path.join(os.path.dirname(__file__), "./texts/long")
    input_filenames = os.listdir(text_dir)

    common_args = dict(
        prompt_template=prompt_template,
        issues_to_summarize=["taxation"],
        output_dir=output_folder,
        min_size=300, max_size=400,
        max_tokens_factor=2.0,
        if_exists='reuse',
        save_summary=True,
        save_log=True,
        dry_run=False
    )

    chunk_args = dict(
        try_no_chunk = False,
        chunk_size = 100000,
        overlap = 2500,
    )
    models = ["gpt-4o-2024-08-06", "claude-3-5-sonnet-20241022", "gemini-1.5-pro-002"]
    for filename in input_filenames:
        for model in models:
            summarize_file(
                os.path.join(text_dir, filename),
                **{**common_args, **chunk_args, "model": model}
            )
    for out_filename in os.listdir(output_folder):
        with open(os.path.join(output_folder, out_filename), "r", encoding="utf-8") as f:
            if out_filename.endswith(".txt"):
                content = f.read()
                assert len(content) > 0
            elif out_filename.endswith(".json"):
                _, model_used, issue_used, file_used = out_filename.split("__")
                content = json.loads(f.read())
                assert len(content["final_summary"]) > 0
                assert len(content["responses"]) == 3
                if model_used.startswith("gpt"):
                    assert model_used == content["responses"][0]["kwargs"]["response_metadata"]["model_name"]
                elif model_used.startswith("claude"):
                    assert model_used == content["responses"][0]["kwargs"]["response_metadata"]["model"]
                elif model_used.startswith("gemini"):
                    assert "prompt_feedback" in content["responses"][0]["kwargs"]["response_metadata"]
                assert content["responses"][0]["kwargs"]["usage_metadata"]["input_tokens"] > 0
                assert content["responses"][0]["kwargs"]["usage_metadata"]["output_tokens"] > 0

@pytest.mark.api
def test_summarize_file_very_long(output_folder, capfd):
    prompt_template = os.path.join(os.path.dirname(__file__), "prompts-summarize__real.yaml")
    text_dir = os.path.join(os.path.dirname(__file__), "./texts/very_long")
    input_filenames = os.listdir(text_dir)

    common_args = dict(
        prompt_template=prompt_template,
        issues_to_summarize=["taxation"],
        output_dir=output_folder,
        min_size=300, max_size=400,
        max_tokens_factor=2.0,
        if_exists='reuse',
        save_summary=True,
        save_log=True,
        dry_run=False
    )

    # Use Chunk
    chunk_args = dict(
        try_no_chunk = True,
        chunk_size = 100000,
        overlap = 2500,
    )

    models = ["gpt-4o-2024-08-06", "claude-3-5-sonnet-20241022"]
    for filename in input_filenames:
        for model in models:
            summarize_file(
                os.path.join(text_dir, filename),
                **{**common_args, **chunk_args, "model": model}
            )

    out, err = capfd.readouterr()
    assert f"Model gpt-4o-2024-08-06 cannot summarize entire document. Trying chunk summarization..." in out
    assert f"Model claude-3-5-sonnet-20241022 cannot summarize entire document. Trying chunk summarization..." in out

    for out_filename in os.listdir(output_folder):
        with open(os.path.join(output_folder, out_filename), "r", encoding="utf-8") as f:
            if out_filename.endswith(".txt"):
                content = f.read()
                assert len(content) > 0
            elif out_filename.endswith(".json"):
                _, model_used, issue_used, file_used = out_filename.split("__")
                content = json.loads(f.read())
                assert len(content["final_summary"]) > 0
                if model_used.startswith("gpt"):
                    assert model_used == content["responses"][0]["kwargs"]["response_metadata"]["model_name"]
                    assert len(content["responses"]) == 10
                elif model_used.startswith("claude"):
                    assert model_used == content["responses"][0]["kwargs"]["response_metadata"]["model"]
                    assert len(content["responses"]) == 10
                elif model_used.startswith("gemini"):
                    assert "prompt_feedback" in content["responses"][0]["kwargs"]["response_metadata"]
                    assert len(content["responses"]) == 1
                assert content["responses"][0]["kwargs"]["usage_metadata"]["input_tokens"] > 0
                assert content["responses"][0]["kwargs"]["usage_metadata"]["output_tokens"] > 0
