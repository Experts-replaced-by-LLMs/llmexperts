import time
from functools import reduce

import anthropic

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage, BaseMessage

from . import openai_model_list, claude_model_list, gemini_model_list, per_minute_token_limit


class LLMClient:

    def __init__(
            self, model, max_tokens,
            temperature=0, max_retries=2, probabilities=False
    ):
        """
        A Wrapper class for various LangChain LLM clients.

        Args:
            model (str): The LLM to use.
            max_tokens (int): The maximum number of tokens.
            temperature (float): The temperature to use.
            max_retries (int): The number of times to retry invoking the model. Defaults to 7, which should be enough
            probabilities: Whether to include token probabilities in the response. Defaults to False. Only works with OpenAI models.
        """
        self.model = model
        if model in openai_model_list:
            self.llm = ChatOpenAI(temperature=temperature, max_tokens=max_tokens, model_name=model, max_retries=max_retries)
        elif model in claude_model_list:
            self.llm = ChatAnthropic(temperature=temperature, max_tokens=max_tokens, model_name=model, max_retries=max_retries)
        elif model in gemini_model_list:
            self.llm = ChatGoogleGenerativeAI(temperature=temperature, max_tokens=max_tokens, model=model, max_retries=max_retries)
        else:
            raise Exception(
                f"You've selected a model that is not available {model}.\nPlease select from the following models: {openai_model_list + claude_model_list + gemini_model_list}"
            )

        # logprobs are only available for OpenAI models
        if probabilities and (model in openai_model_list):
            self.llm = self.llm.bind(logprobs=True)
        elif probabilities:
            print(
                f"Probabilities are not available for model {model}, please select a model from the following list: {openai_model_list}")

        self.tokens_used = 0
        self.token_limit = per_minute_token_limit.get(model, 800000)
        self.start_time = time.time()

    def bind(self, **kwargs):
        self.llm = self.llm.bind(**kwargs)

    def wait_for_per_minute_limit(self, prompt):
        char_length = reduce(lambda x, y: len(x.content) + len(y.content), prompt)
        if self.tokens_used + char_length / 4 > self.token_limit:
            elapsed_time = time.time() - self.start_time
            time_to_wait = 60 - elapsed_time
            if time_to_wait > 0:
                print(f'Waiting for {time_to_wait:.0f} seconds to avoid token limit. Tokens used: {self.tokens_used}')
                time.sleep(time_to_wait)
            self.tokens_used = 0
            self.start_time = time.time()

    def mock_response(self, prompt, max_chars=2000, prefix="MOCK CONTENT", response_content=None):

        if response_content is not None:
            mock_content = response_content
        else:
            if isinstance(prompt, str):
                mock_content = prompt
            elif isinstance(prompt, BaseMessage):
                mock_content = prompt.content
            elif isinstance(prompt, list):
                mock_content = prompt[-1].content
            else:
                mock_content = ""

            if len(mock_content) > max_chars:
                mock_content = mock_content[:int(max_chars/2)]+ " ... "+ mock_content[-int(max_chars/2):]
            mock_content = f"{prefix} {mock_content}"

        mock = AIMessage(
            content=mock_content, additional_kwargs={'refusal': None},
            response_metadata={},
            usage_metadata={'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
        )
        return mock

    def invoke(self, prompt, dry_run=False, dry_run_res=None):
        """
        Invoke the LangChain LLM client.

        Args:
            prompt: A list of Messages. Could be HumanMessage, SystemMessage or AI Message.
            dry_run (bool|str): Do not invoke the LLMs. Return a mock response instead. For testing purposes.
                If a str is given, prefix this str to the response.
            dry_run_res (str): Content to use for the dry run mock response.

        Returns:
            LangChain's Response.

        """
        if isinstance(dry_run, str) or dry_run == True:
            prefix = dry_run if isinstance(dry_run, str) else ""
            return self.mock_response(prompt, prefix=prefix, response_content=dry_run_res)
        self.wait_for_per_minute_limit(prompt)
        try:
            response = self.llm.invoke(prompt)
        except anthropic.RateLimitError:
            print("Anthropic rate limit exceeded. Waiting for 1 minute ...")
            time.sleep(60)
            print("Retry Anthropic invoke.")
            response = self.llm.invoke(prompt)
        try:
            self.tokens_used += response.response_metadata['token_usage']['prompt_tokens']
        except:
            pass
        return response

    def batch(self, prompt_batch, dry_run=False, dry_run_res=None):
        """
        Invoke LLMs using LangChain's batch method.
        Args:
            prompt_batch: A batch of list of Messages.
            dry_run (bool|str): Do not invoke the LLMs. Return a mock response instead. For testing purposes.
                If a str is given, prefix this str to the response.
            dry_run_res (str): Content to use for the dry run mock response.

        Returns:
            A list of LangChain's Responses.

        """

        if isinstance(dry_run, str) or dry_run==True:
            prefix = dry_run if isinstance(dry_run, str) else ""
            return [self.mock_response(p, prefix=prefix, response_content=dry_run_res) for p in prompt_batch]
        return self.llm.batch(prompt_batch)
