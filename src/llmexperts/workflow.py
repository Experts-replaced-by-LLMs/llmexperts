import os

from .analyze import analyze_file
from .prompts import SummarizePromptTemplate, AnalyzePromptTemplate
from .summarize import summarize_file


def analyze_workflow(
        filepaths, output_dir,
        summary_prompt_template: SummarizePromptTemplate | os.PathLike | str,
        analyze_prompt_template: AnalyzePromptTemplate | os.PathLike | str,
        issues_to_summarize, summarize_models, analyze_models, summary_all_issue=False,
        analyze_result_filename="analyze_results.csv",
        chunk_size=100000, overlap=2500, min_size=500, max_size=1000, max_tokens_factor=1.0,
        parse_retries=3, max_retries=7, concurrency=3, probabilities=False,
        ues_examples=False, personas=None, encouragements=None
):
    # Make output directories
    summary_output_dir = os.path.join(output_dir, "summary")
    if not os.path.exists(summary_output_dir):
        os.makedirs(summary_output_dir)
    summary_log_dir = os.path.join(output_dir, "summary_log")
    if not os.path.exists(summary_log_dir):
        os.makedirs(summary_log_dir)

    # Summarize documents
    for filepath in filepaths:
        for summary_model in summarize_models:
            if summary_all_issue:
                summarize_file(
                    filepath, summary_prompt_template,
                    issues_to_summarize, summary_output_dir, summary_model,
                    chunk_size=chunk_size, overlap=overlap, min_size=min_size, max_size=max_size,
                    max_tokens_factor=max_tokens_factor,
                    save_log=True, log_dir=summary_log_dir, dry_run=True
                )
            else:
                for issue in issues_to_summarize:
                    summarize_file(
                        filepath, summary_prompt_template, [issue],
                        summary_output_dir, summary_model, save_log=True, dry_run=True,
                        log_dir=summary_log_dir
                    )

    # Analyze summaries
    analyze_result_filepath = os.path.join(output_dir, analyze_result_filename)

    for filename in os.listdir(summary_output_dir):
        if not filename.startswith("summary") or not filename.endswith(".txt"):
            continue
        filepath = os.path.join(summary_output_dir, filename)
        _, summarize_model_name, summarize_issue, summarize_filename = filename.split("__")
        issues_to_analyze = issues_to_summarize if summary_all_issue else [summarize_issue]
        analyze_file(
            filepath, analyze_models, issues_to_analyze, analyze_prompt_template, output_dir,
            parse_retries=parse_retries, max_retries=max_retries, concurrency=concurrency, probabilities=probabilities,
            use_examples=ues_examples, override_personas=personas, override_encouragements=encouragements, dry_run=True,
            results_filepath=analyze_result_filepath,
            meta_columns={"summarize_model": summarize_model_name, "original_file": summarize_filename},
            skip_existing_analyze_results=True
        )
