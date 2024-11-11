import os
from dotenv import load_dotenv
from src.llmexperts.summarize import summarize_file

load_dotenv()

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
    log_dir = "./test_output/summary_logs"

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


# def test_summarize_file_chunks():
#     # try no chunk with chunk
#     # try no chunk without chunk
#     # no try no chunk with chunk
#     # no try no chunk without chunk
#     pass

def test_summarize_file(output_folder, short_text_file_folder):
    prompt_template = os.path.join(os.path.dirname(__file__), "prompts-summarize.yaml")
    input_filenames = os.listdir(short_text_file_folder)
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
        dry_run=False
    )

    issues_to_summarize = ["issue_1", "issue_2"]
    models = ["gpt-4o-2024-08-06", "claude-3-5-sonnet-20241022", "gemini-1.5-pro-002"]

    for filename in input_filenames:
        for model in models:
            for iss in issues_to_summarize:
                summarize_file(
                    os.path.join(short_text_file_folder, filename),
                    **{**common_args, "model": model, "issues_to_summarize": [iss]}
                )

    output_filenames = os.listdir(output_folder)

    assert len(output_filenames) == len(input_filenames) * len(models) * len(issues_to_summarize) * 2
    for model in models:
        for filename in input_filenames:
            for iss in issues_to_summarize:
                assert os.path.isfile(
                    os.path.join(output_folder, f"summary_standard__{model}__{iss}__{filename}")) == True
                assert os.path.isfile(os.path.join(output_folder,
                                                   f"log_summary_standard__{model}__{iss}__{os.path.splitext(filename)[0]}.json")) == True
