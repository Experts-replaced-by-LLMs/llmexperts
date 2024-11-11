# llmexperts

Python package for processing political texts for summarsiation and scaling of the texts as political experts would otherwise do.

## Overview

The llmexperts repository aims to provide a comprehensive framework for analyzing and evaluating political manifestos. By utilizing a set of predefined political scales, and a range of LLMs users can objectively assess the positions and ideologies presented in various political documents.

## Features

- **Summarization**: The repository offers a summarizer that uses LLMs to summarize political manifesto documents of various languages into English.
- **Scoring System**: The repository offers a scoring system that uses LLMs to assign numerical values to different aspects of political manifestos based on predefined scales.
- **Customizable Scales**: Users can modify this code to create and customize their own scales to suit their specific analysis requirements.

## Prompts

Both summarization and scoring requires prompts to work with the LLMs. The prompts should be specified in .yml files with some predefined fields.

The variables wrapped in brackets denote variables that will be substituted in run time.

### Summarization prompts

#### issue_areas

A dictionary whose keys are issue names and values are issue definitions.

#### **system_template_string**

A template string for system message.

**{issue_areas}**: Must be included. This will be replaced by issue definition in run time.

**{min_size}**, **{max_size}**: Optional range of summarization length.

**{text}**: Must be included. This will be replaced by the text to be summarized.

Example:

```yaml
issue_areas:
  issue_1: |
    ISSUE_1 DEFINITION
  issue_2: |
    ISSUE_2 DEFINITION
system_template_string: |
  SUMMARY {issue_areas} IN {min_size} - {max_size} words.
human_template_string: |
  Summarize: {text}
```

### Scoring prompts

#### personas, encouragements

Personas and encouragements to add to the system message.

#### policy_scales:

The definition of policy scales.

### examples:

The examples of scoring, used for few-shot scoring.

```yaml
personas:
  - PERSONA_1
  - PERSONA_2
  - PERSONA_3

encouragements:
  - ENCOURAGEMENT_1
  - ENCOURAGEMENT_2
  - ENCOURAGEMENT_3

policy_scales:
  issue_1: |
    ISSUE_1 DEFINITION
  issue_2: |
    ISSUE_2 DEFINITION

system_template_string: |
  {persona} {encouragement}

  {policy_scale}

human_template_string: |
  Analyze the following political text:

  {text}

examples:
  issue_1:
    - score: 1
      summary: "ISSUE 1 EXAMPLE 1"
    - score: 7
      summary: 'ISSUE 1 EXAMPLE 2'
  issue_2:
    - score: 2
      summary: "ISSUE 2 EXAMPLE 1"
    - score: 3
      summary: 'ISSUE 2 EXAMPLE 2'
```

## API Keys

To use the LLM APIs, the LLM's API keys needs to be set as environment variables:

```bash
$ export OPENAI_API_KEY=<your_openai_api_key>
$ export ANTHROPIC_API_KEY=<your_anthropic_api_key>
$ export GOOGLE_API_KEY=<your_google_api_key>
```

## Summarization

```python
from llmexperts.summarize import summarize_file

filepath = "./texts/manifesto_text.txt"
prompt_template = "./prompts-summarize.yaml"
output_dir = "./output"
log_dir = "./logs"

issues_to_summarize = ["issue_1", "issue_2"]
model = "gpt-4o-2024-08-06"

summarize_file(
    filepath, prompt_template, issues_to_summarize, output_dir, model, try_no_chunk=False,
    save_log=False, log_dir=log_dir
)
``` 

To summarize each issue and save in a separate summary file:

```python
for issue in issues_to_summarize:
    summarize_file(
        filepath, prompt_template, [issue], output_dir, model, try_no_chunk=False,
        save_log=False, log_dir=log_dir
    )
```

## Scoring

```python
from llmexperts.analyze import analyze_file

filepath = "./texts/manifesto_text.txt"
prompt_template = "./prompts-analyze.yaml"
output_dir = "./output"
log_dir = "./logs"

model_list = ["gpt-4o-2024-08-06", "claude-3-5-sonnet-20241022", "gemini-1.5-pro-002"]
issue_list = ["issue_1", "issue_2"]

analyze_file(
    filepath, model_list, issue_list, prompt_template, output_dir,
    use_examples=True ,save_log=True, logs_filepath=log_dir
)
```

For more details, please see documentation.

## Test

To run the tests, make sure the LLM's API key are properly set as environment variables.

```bash
$ export OPENAI_API_KEY=<your_openai_api_key>
$ export ANTHROPIC_API_KEY=<your_anthropic_api_key>
$ export GOOGLE_API_KEY=<your_google_api_key>
```

Alternatively, put the variables in a .env file in the /tests folder, which will then be loaded by `python-dotenv` package.

Run the tests:

```bash
$ tox run -e test
```
