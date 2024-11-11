import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

# Tested models, this list should be updated as new models are added
# Each service requires an API key to work, stored as an envorinment variable

# Checked models:
# - OpenAI models: ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4']
# - Claude models: ['claude-3-5-sonnet-20240620', 'claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307']
# - Gemini models: ['gemini-1.5-pro-001']

openai_model_list = [
    'gpt-3.5-turbo', 'gpt-4', 'gpt-4o-2024-08-06'
]
claude_model_list = [
    "claude-3-5-sonnet-20241022", 'claude-3-5-sonnet-20240620',
    'claude-3-opus-20240229',
    'claude-3-sonnet-20240229',
    'claude-3-haiku-20240307'
]
gemini_model_list = [
    'gemini-1.5-pro-001',
    'gemini-1.5-pro-002'
]

per_minute_token_limit = {
    "claude-3-5-sonnet-20240620": 400000,
    "claude-3-opus-20240229": 80000,
    "claude-3-sonnet-20240229": 160000,
    "claude-3-haiku-20240307": 200000,
    "gpt-4o": 800000
}
