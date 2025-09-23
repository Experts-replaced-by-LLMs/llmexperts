import json
import os
# import re
# from os import PathLike

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.load import dumpd

from .model import LLMClient
from .prompts import SummarizePromptTemplate
from .utils import escape_model_name


class Summary:

    def __init__(self, final_summary: str, responses):
        self.final_summary = final_summary
        self.responses = responses

    def __repr__(self):
        return self.final_summary

    def dump(self):
        return json.dumps({
            "responses": [dumpd(r) for r in self.responses],
            "final_summary": self.final_summary,
        })


def summarize_text(
        text, prompt_template: SummarizePromptTemplate | os.PathLike, model, issues_to_summarize,
        chunk_size=100000, overlap=2500, max_tokens_factor=1.0,
        min_size=500, max_size=1000, debug=False, dry_run=False
) -> Summary:
    """
    Summarizes the given text based on the specified issue areas using a language model.
    This approach creates all the summaries in one pass to avoid repeating the summarization process.

    Args:
        text (str): The text to be summarized.
        prompt_template (SummarizePromptTemplate|PathLike): A SummarizePrompt instance or a filepath to the prompt file.
        model (str, optional): The name of the language model to be used for summarization. Defaults to "gpt-4o".
        issues_to_summarize (list): The issues to be summarized.
        chunk_size (int, optional): The size of each chunk to split the text into. Defaults to 100000. Set to 0 to disable chunk.
        overlap (int, optional): The overlap between consecutive chunks. Defaults to 2500.
        min_size (int, optional): The minimum size of the final summary. Defaults to 500.
        max_size (int, optional): The maximum size of the final summary. Defaults to 1000.
        max_tokens_factor (float, optional): The max_tokens of LLM will be set to summary_size[1]*max_tokens_factor
        debug (bool, optional): Should debug information be printed. Defaults to False.
        dry_run (bool, optional): Don't invoke the LLM api call. Return a mock response for debug and testing. Defaults to False.

    Returns:
        Summary: The final summary of the text.

    """

    if not isinstance(prompt_template, SummarizePromptTemplate):
        prompt_template = SummarizePromptTemplate.from_file(prompt_template)

    # system_kwargs = {"max_size": max_size, "min_size": min_size}

    if chunk_size > 0:
        if chunk_size < 1:
            chunk_size = int(len(text)*chunk_size)
        # Split the text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        chunks = text_splitter.split_text(text)
    else:
        chunks = [text]

    # Setup the LLM
    max_tokens = max_size * max_tokens_factor
    llm = LLMClient(model, max_tokens, temperature=0)

    # Summarize each chunk
    responses = []

    print(f"Using {model} for summarization.")

    for chunk in chunks:
        summarize_prompt = prompt_template.build_prompt(
            chunk, issues_to_summarize, min_size=min_size, max_size=max_size
        )

        if debug:
            print('Prompt:', summarize_prompt)

        if dry_run:
            dry_run = f"[MOCK SUMMARIZE][{' '.join(issues_to_summarize)}][{model}]"

        summary_response = llm.invoke(summarize_prompt, dry_run=dry_run)
        responses.append(summary_response)

        print(f'Summarized so far: {len(responses)} out of {len(chunks)} chunks', end='\r')
    print('\n', end='\r')

    # Combine all summaries into one final summary
    if len(responses) > 1:
        print('Combining summaries into one final summary')
        final_summaries = " ".join([s.content for s in responses])
        final_summarize_prompt = prompt_template.build_prompt(
            final_summaries, issues_to_summarize, min_size=min_size, max_size=max_size
        )
        final_summary_response = llm.invoke(final_summarize_prompt, dry_run=dry_run)
        responses.append(final_summary_response)

    final_summary = responses[-1].content
    print(f'Final summary length: {len(final_summary)} characters \n')
    return Summary(final_summary, responses)


def make_summary_name(max_size, model_name, issue_areas, file_path):

    input_filename, _ = os.path.splitext(os.path.basename(file_path))

    if max_size == 1000:
        length = 'standard'
    elif max_size < 1000:
        length = 'short'
    else:
        length = 'long'
    # formatted_model_name = re.sub(r'[-_.:]', '', model_name)
    formatted_model_name = escape_model_name(model_name)
    if len(issue_areas) > 1:
        issue_name = "multi"
    else:
        issue_name = issue_areas[0]
    return f"summary_{length}__{formatted_model_name}__{issue_name}__{input_filename}"


def summarize_file(
        file_path, prompt_template: SummarizePromptTemplate | os.PathLike | str,
        issues_to_summarize, output_dir, model, try_no_chunk=False,
        chunk_size=100000, overlap=2500, min_size=500, max_size=1000, max_tokens_factor=1.0,
        if_exists='reuse', save_summary=True,  save_log=False, log_dir=None,
        debug=False, dry_run=False
) -> str:
    """
    Summarizes the text in the given file based on the specified issue area using a language model.

    Args:
        file_path (str): The path to the file containing the text to be summarized.
        prompt_template (SummarizePromptTemplate|PathLike): A SummarizePrompt instance or a filepath to the prompt file.
        issues_to_summarize (list): The issue areas related to the text.
        output_dir (str): The path to the directory where the results will be stored. Default to "../data/summaries/".
        model (str): The name of the language model to be used for summarization.
        try_no_chunk (bool, optional): Try to summarize without chunking first. Fall back to chunking when exception raised.
        chunk_size (int, optional): The size of each chunk to split the text into. Defaults to 100000.
        overlap (int, optional): The overlap between consecutive chunks. Defaults to 2500.
        min_size (int, optional): The minimum size of the final summary. Defaults to 500.
        max_size (int, optional): The maximum size of the final summary. Defaults to 1000.
        max_tokens_factor (float, optional): The max_tokens of LLM will be set to summary_size[1]*max_tokens_factor
        if_exists (str, optional): What to do if the summary file already exists. Options are 'overwrite', 'reuse' Defaults to 'overwrite'
        save_summary (bool, optional): Should the summary be saved to a file. Defaults to True.
        save_log (bool, optional): Should the log information be saved to a file. Defaults to False.
        log_dir (str, optional): The path to the directory where the summary logs will be stored. If None and save_summary is True, save logs to the summary output dir.
        debug (bool, optional): Should debug information be printed. Defaults to False.
        dry_run (bool, optional): Don't invoke the LLM api call. Return a mock response for debug and testing. Defaults to False.

    Returns:
        str: The final summary of the text.

    """

    # Create output directories
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if save_log and log_dir is not None and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Handle single issues gracefully
    if isinstance(issues_to_summarize, str):
        issues_to_summarize = [issues_to_summarize]

    # Constructs the output file name
    summary_name = make_summary_name(max_size, model, issues_to_summarize, file_path)

    summary_file_name = os.path.join(
        output_dir, f"{summary_name}.txt"
    )

    # Check if the summary file already exists, and reuses it if requested, exiting early
    if if_exists == 'reuse':
        if os.path.exists(summary_file_name):
            print(f"Summary file {os.path.basename(summary_file_name)} already exists. Reusing the existing summary.")
            with open(summary_file_name, "r", encoding="utf-8") as file:
                return file.read()

    # Main summarization process
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    if try_no_chunk:
        try:
            print("Trying summarize without chunk ...")
            summary = summarize_text(
                text, prompt_template, model, issues_to_summarize,
                chunk_size=0, overlap=0, max_tokens_factor=max_tokens_factor,
                min_size=min_size, max_size=max_size, debug=debug, dry_run=dry_run
            )
        except Exception as e:
            if chunk_size > 0:
                print(f"Model {model} cannot summarize entire document. Trying chunk summarization...")
                summary = summarize_text(
                    text, prompt_template, model, issues_to_summarize,
                    chunk_size=chunk_size, overlap=overlap, max_tokens_factor=max_tokens_factor,
                    min_size=min_size, max_size=max_size, debug=debug, dry_run=dry_run
                )
            else:
                raise e
    else:
        summary = summarize_text(
            text, prompt_template, model, issues_to_summarize,
            chunk_size=chunk_size, overlap=overlap, max_tokens_factor=max_tokens_factor,
            min_size=min_size, max_size=max_size, debug=debug, dry_run=dry_run
        )

    if save_log:
        if log_dir is not None:
            log_file_name = os.path.join(
                log_dir, f"log_{summary_name}.json"
            )
        else:
            log_file_name = os.path.join(
                output_dir, f"log_{summary_name}.json"
            )
        with open(log_file_name, 'a', encoding="utf-8") as f:
            f.write(summary.dump())
            f.write('\n')
    if save_summary:
        print(f"Saving summary to {summary_file_name}")
        with open(summary_file_name, "w", encoding="utf-8") as file:
            file.write(summary.final_summary)

    return summary.final_summary
