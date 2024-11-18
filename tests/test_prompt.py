import itertools
import os

import pytest

from src.llmexperts.prompts import ScalePromptTemplate, SummarizePromptTemplate


def test_summarize_prompt():
    prompt_template=os.path.join(os.path.dirname(__file__), "prompts-summarize.yaml")
    test_text = "TEST TEXT"
    test_issue = "issue_1"
    min_size=300
    max_size=400
    summarize_prompt_template = SummarizePromptTemplate.from_file(prompt_template)
    assert type(summarize_prompt_template.issue_areas) == dict
    prompt = summarize_prompt_template.build_prompt(test_text, [test_issue], max_size=max_size, min_size=min_size)
    assert prompt[0].content == f"SUMMARY 1. {test_issue}: {summarize_prompt_template.issue_areas[test_issue]} IN {min_size} - {max_size} words.\n"
    assert prompt[1].content == f"Summarize: {test_text}"

def test_scale_prompt():
    prompt_template = os.path.join(os.path.dirname(__file__), "prompts-scale.yaml")
    test_text = "TEST TEXT"
    test_issue = "issue_1"
    scale_prompt_template = ScalePromptTemplate.from_file(prompt_template)
    assert type(scale_prompt_template.policy_scales) == dict

    policy_scale = scale_prompt_template.policy_scales[test_issue]

    prompt = scale_prompt_template.build_prompt(test_text, test_issue, override_persona_to_use=0, override_encouragement_to_use=1)
    assert len(prompt) == 1
    assert prompt[0].prompt[0].content == f"{scale_prompt_template.personas[0]} {scale_prompt_template.encouragements[1]}\n\n{policy_scale}\n"
    assert prompt[0].prompt[1].content == f"Scale the following political text:\n\n{test_text}\n"

    prompts = scale_prompt_template.build_prompt(test_text, test_issue)
    assert len(prompts) == 9
    combinations = itertools.product(scale_prompt_template.personas, scale_prompt_template.encouragements)
    for p, c in zip(prompts, combinations):
        assert p.prompt[0].content == f"{c[0]} {c[1]}\n\n{policy_scale}\n"
        assert p.prompt[1].content == f"Scale the following political text:\n\n{test_text}\n"

    prompts = scale_prompt_template.build_prompt(test_text, test_issue, override_persona_to_use=[0, 1, 2], override_encouragement_to_use=[0, 2])
    combinations = itertools.product([scale_prompt_template.personas[pi] for pi in [0,1,2]], [scale_prompt_template.encouragements[ei] for ei in [0,2]])
    assert len(prompts) == 6
    for p, c in zip(prompts, combinations):
        assert p.prompt[0].content == f"{c[0]} {c[1]}\n\n{policy_scale}\n"
        assert p.prompt[1].content == f"Scale the following political text:\n\n{test_text}\n"

def test_scale_prompt_with_examples():
    prompt_template = os.path.join(os.path.dirname(__file__), "prompts-scale__examples.yaml")
    test_text = "TEST TEXT"
    test_issue = "issue_2"

    scale_prompt_template = ScalePromptTemplate.from_file(prompt_template)
    assert type(scale_prompt_template.examples) == dict
    prompts = scale_prompt_template.build_prompt(test_text, test_issue, use_examples=True)
    assert len(prompts) == 9
    for p in prompts:
        assert len(p.prompt) == 2*2+2

    prompts = scale_prompt_template.build_prompt(test_text, test_issue, use_examples=False)
    assert len(prompts) == 9
    for p in prompts:
        assert len(p.prompt) == 2

    prompt_template = os.path.join(os.path.dirname(__file__), "prompts-scale.yaml")
    scale_prompt_template = ScalePromptTemplate.from_file(prompt_template)
    assert type(scale_prompt_template.examples) == dict
    prompts = scale_prompt_template.build_prompt(test_text, test_issue, use_examples=True)
    assert len(prompts) == 9
    for p in prompts:
        assert len(p.prompt) == 2

    prompts = scale_prompt_template.build_prompt(test_text, test_issue, use_examples=False)
    assert len(prompts) == 9
    for p in prompts:
        assert len(p.prompt) == 2
