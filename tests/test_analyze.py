import os
import pytest
import pandas as pd
from dotenv import load_dotenv

from src.llmexperts.analyze import analyze_file, ensure_output_paths

load_dotenv()


def test_ensure_output_paths(output_folder):

    results_filepath = os.path.join(output_folder, "./output/results.csv")
    ensure_output_paths(
        results_filepath=results_filepath,
    )
    assert os.path.exists(os.path.dirname(results_filepath))

    with pytest.raises(ValueError):
        ensure_output_paths(results_filepath="./output/results")


def test_analyze_file(output_folder, summary_file_folder):
    """
    examples = [True, False]
    personas = [None, 0, [0,1]]
    encouragement = [None, 0, [0,1]]
    dry_run = True
    meta_columns = [None, {}, {1col}, {2col}]
    skip_existing_analyze_res = [True, False]
    """
    model_list = ["gpt-4o-2024-08-06", "claude-3-5-sonnet-20241022", "gemini-1.5-pro-002"]
    issue_list = ["issue_1", "issue_2"]
    prompt_template=os.path.join(os.path.dirname(__file__), "prompts-analyze.yaml")
    args = dict(
        model_list=model_list,
        issue_list=issue_list,
        prompt_template=prompt_template,
        output_dir=output_folder,
        use_examples=False,
        override_personas=None, override_encouragements=None,
        parse_retries = 3, max_retries = 7, concurrency = 3, probabilities = False,
        results_filepath = None, results_filename = "analyze_results.csv",
        save_log = True,
        meta_columns = None, skip_existing_analyze_results = True,
        dry_run=True
    )

    filenames = os.listdir(summary_file_folder)

    for filename in filenames:
        file_path = os.path.join(summary_file_folder, filename)
        analyze_file(file_path, **args)

    df = pd.read_csv(os.path.join(output_folder, "analyze_results.csv"))
    assert df.shape[0] == len(filenames)*len(model_list)*9*len(issue_list)

    # Test skip existing
    for filename in filenames:
        file_path = os.path.join(summary_file_folder, filename)
        analyze_file(file_path, **args)

    df = pd.read_csv(os.path.join(output_folder, "analyze_results.csv"))
    assert df.shape[0] == len(filenames)*len(model_list)*9*len(issue_list)


def test_analyze_file_use_examples(output_folder, summary_file_folder):
    model_list = ["gpt-4o-2024-08-06", "claude-3-5-sonnet-20241022", "gemini-1.5-pro-002"]
    issue_list = ["issue_1", "issue_2"]
    prompt_template = os.path.join(os.path.dirname(__file__), "prompts-analyze.yaml")
    args = dict(
        model_list=model_list,
        issue_list=issue_list,
        prompt_template=prompt_template,
        output_dir=output_folder,
        use_examples=True,
        override_personas=0, override_encouragements=1,
        parse_retries = 3, max_retries = 7, concurrency = 3, probabilities = False,
        results_filepath = None, results_filename = "analyze_results.csv",
        save_log = True,
        meta_columns = None, skip_existing_analyze_results = True,
        dry_run=True
    )

    filenames = os.listdir(summary_file_folder)

    for filename in filenames:
        file_path = os.path.join(summary_file_folder, filename)
        analyze_file(file_path, **args)

    df = pd.read_csv(os.path.join(output_folder, "analyze_results.csv"))
    assert df.shape[0] == len(filenames)*len(model_list)*len(issue_list)

    # Test skip existing
    for filename in filenames:
        file_path = os.path.join(summary_file_folder, filename)
        analyze_file(file_path, **args)

    df = pd.read_csv(os.path.join(output_folder, "analyze_results.csv"))
    assert df.shape[0] == len(filenames)*len(model_list)*len(issue_list)
