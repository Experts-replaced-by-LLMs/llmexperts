import json
import tiktoken
import anthropic
import yaml

from vertexai.preview import tokenization


def count_tokens(text, model):
    if model == "gpt":
        encoding = tiktoken.encoding_for_model('gpt-4')
        enc = encoding.encode(text)
        return len(enc)
    elif model == "claude":
        ac = anthropic.Client()
        return ac.count_tokens(text)
    elif model == "gemini":
        model_name = "gemini-1.5-pro-001"
        tokenizer = tokenization.get_tokenizer_for_model(model_name)
        result = tokenizer.count_tokens(text)
        return result.total_tokens


def yml_to_dict(filepath):
    """
    Read yaml file

    Args:
        filepath: The file path

    Returns: Dict of the yaml file content

    """
    with open(filepath, "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f.read())

    return prompts


def json_to_dict(filepath):
    """
    Read json file

    Args:
        filepath: The file path

    Returns: Dict of the json file content

    """
    with open(filepath, "r", encoding="utf-8") as f:
        prompts = json.loads(f.read())

    return prompts