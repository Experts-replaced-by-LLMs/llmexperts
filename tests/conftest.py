import os
import shutil

import pytest

@pytest.fixture(scope='function')
def output_folder():
    path = "./test_output"
    if not os.path.exists(path):
        os.mkdir(path)
    yield path
    # shutil.rmtree(path)

@pytest.fixture(scope='function')
def text_file_folder():
    path = "./test_texts"
    if not os.path.exists(path):
        os.mkdir(path)
    for i in range(2):
        with open(os.path.join(path, "text_" + str(i) + ".txt"), "w", encoding="utf-8") as f:

            f.write(f"TEXT {i}")
    yield path
    # shutil.rmtree(path)


@pytest.fixture(scope='function')
def short_text_file_folder():
    path = "./test_texts_long"
    if not os.path.exists(path):
        os.mkdir(path)
    with open(os.path.join(path, "text_1.txt"), "w", encoding="utf-8") as f:
        f.write(" ".join(["TEXT"]*100))
    with open(os.path.join(path, "text_2.txt"), "w", encoding="utf-8") as f:
        f.write(" ".join(["TEXT"]*1_000))
    yield path
    # shutil.rmtree(path)


@pytest.fixture(scope='function')
def long_text_file_folder():
    path = "./test_texts_long"
    if not os.path.exists(path):
        os.mkdir(path)
    with open(os.path.join(path, "text_1.txt"), "w", encoding="utf-8") as f:
        f.write(" ".join(["TEXT"]*100_000))
    with open(os.path.join(path, "text_2.txt"), "w", encoding="utf-8") as f:
        f.write(" ".join(["TEXT"]*250_000))
    yield path
    # shutil.rmtree(path)


@pytest.fixture(scope='function')
def summary_file_folder():
    path = "./test_summary"
    issues_to_summarize = ["issue_1", "issue_2"]
    models = ["gpt-4o-2024-08-06", "claude-3-5-sonnet-20241022", "gemini-1.5-pro-002"]
    input_filenames = ["text_1.txt", "text_2.txt"]
    if not os.path.exists(path):
        os.mkdir(path)

    for model in models:
        for filename in input_filenames:
            for iss in issues_to_summarize:
                p = os.path.join(path, f"summary_standard__{model}__{iss}__{filename}")
                with open(p, "w", encoding="utf-8") as f:
                    f.write(f"SUMMARY OF {path}")
    yield path
    shutil.rmtree(path)

