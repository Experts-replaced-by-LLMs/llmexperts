import abc
import os
from collections import namedtuple
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage, AIMessage, BaseMessage
from .utils import yml_to_dict, json_to_dict

class LLMExpertPromptTemplate:

    def __init__(self, system_template_string, human_template_string):
        """
        Prompt template

        Args:
            system_template_string: Template string to build system message
            human_template_string: Template to build human message
        """
        self.system_template_string = system_template_string
        self.human_template_string = human_template_string
        self.system_template = PromptTemplate(template=system_template_string)
        self.human_template = PromptTemplate(template=human_template_string)

    @classmethod
    def from_file(cls, filepath: str|os.PathLike):
        """
        Load template from external file. Currently support yaml and json format.
        Args:
            filepath: The path to the template file.

        Returns:
            LLMExpertPromptTemplate

        """
        _, ext = os.path.splitext(filepath)
        if ext in [".yml", ".yaml"]:
            prompt_args = yml_to_dict(filepath)
        elif ext == ".json":
            prompt_args = json_to_dict(filepath)
        else:
            raise NotImplementedError(f"File type {ext} not supported.")
        return cls(**prompt_args)

    @abc.abstractmethod
    def build_prompt(self, *args, **kwargs):
        pass

class SummarizePromptTemplate(LLMExpertPromptTemplate):

    def __init__(self, system_template_string, human_template_string, issue_areas):
        """
        Prompt template for summarization.

        Args:
            system_template_string: Template string to build system message
            human_template_string: Template to build human message
            issue_areas (dict): The definition of issue areas. Should be a dictionary where keys are issue names and values are issue definition.
        """
        super().__init__(system_template_string, human_template_string)
        self.issue_areas = issue_areas

    def build_prompt(self, text, issues_to_summarize, min_size=500, max_size=1000):
        """
        Build a summarization prompt.

        Args:
            text (str): The text to be summarized.
            issues_to_summarize (list[str]): A list of issue names to be summarized. Name must be in the template otherwise an Exception will be raised.
            min_size (int): The minimum size of the summarization output.
            max_size: (int): The maximum size of the summarization output.

        Returns:
            A list with a SystemMessage and a HumanMessage.

        """

        issue_area_dict = {issue: self.issue_areas[issue] for issue in issues_to_summarize}
        issue_area_descriptions = [f"{issue}: {description}" for issue, description in issue_area_dict.items()]
        issue_list_string = "\n".join([f"{i + 1}. {area}" for i, area in enumerate(issue_area_descriptions)])

        return [
            SystemMessage(
                content=self.system_template.format(
                    issue_areas=issue_list_string,
                    min_size=min_size,
                    max_size=max_size
                )),
            HumanMessage(
                content=self.human_template.format(text=text)
            )
        ]

ScalePrompt = namedtuple("ScalePrompt", ["prompt", "persona", "encouragement", "persona_idx", "encouragement_idx"])

class ScalePromptTemplate(LLMExpertPromptTemplate):

    def __init__(
            self, system_template_string: str, human_template_string: str,
            policy_scales: dict, personas: list[str], encouragements: list[str],
            examples: dict[str, list[dict[str, str|int]]]=None, ai_template_string: str = "{score}"
    ):
        """
        Prompt template for scoring.

        Args:
            system_template_string: Template string to build system message
            human_template_string: Template to build human message
            policy_scales (dict): Definition of policy scales.
            personas (list[str]): A list of personas.
            encouragements (list[str]): A list of encouragements.
            examples (dict): Examples of policy scales. Used for few-shot scoring. Should be a dictionary where keys are policy names and values are a dict containing a "score" and a "summary" key.
            ai_template_string (str): Template string to build AI message. Used for few-shot scoring.
        """
        super().__init__(system_template_string, human_template_string)
        self.policy_scales = policy_scales
        self.personas = personas
        self.encouragements = encouragements
        self.examples = examples or {}
        self.ai_template_string = ai_template_string
        self.ai_template = PromptTemplate(template=ai_template_string)

    def build_prompt(
            self, text: str, issue_to_scale: str, use_examples:bool=False,
            override_persona_to_use: int | list[int]=None, override_encouragement_to_use: int | list[int]=None
    ) -> list[ScalePrompt]:

        """
        Build a scoring prompt.

        Args:
            text (str): The text to be scored.
            issue_to_scale (str): The issue name to be scored.
            use_examples (boolean): Whether to add examples as few-shot scoring.
            override_persona_to_use (int|list[int]): The persona to use. The index of personas in the template. If None, use all.
            override_encouragement_to_use: (int|list[int]): The encouragement to use. The index of encouragements in the template. If None, use all.

        Returns:
            A two-dimensional list of shape (n_persona * n_encouragement, 2 + 2*n_examples).
            Dim 0 is the combinations of persona and encouragement.
            Dim 1 is prompts for each combination containing a System message and a HumanMessage. Plus many (HumanMessage, AI message) pairs in between.

        """

        if override_persona_to_use is None:
            override_persona_to_use = list(range(len(self.personas)))
        elif type(override_persona_to_use) is int:
            override_persona_to_use = [override_persona_to_use]
        if override_encouragement_to_use is None:
            override_encouragement_to_use = list(range(len(self.encouragements)))
        if type(override_encouragement_to_use) is int:
            override_encouragement_to_use = [override_encouragement_to_use]

        example_messages = []
        if use_examples:
            for issue_examples in self.examples.get(issue_to_scale, []):
                summary = issue_examples["summary"]
                score = issue_examples["score"]
                example_messages.append(HumanMessage(content=self.human_template.format(text=summary)))
                example_messages.append(AIMessage(content=self.ai_template.format(score=f"{score}")))

        prompts = []
        for persona_idx in override_persona_to_use:
            persona_text = self.personas[persona_idx]
            for encouragement_idx in override_encouragement_to_use:
                encouragement_text = self.encouragements[encouragement_idx]
                prompts.append(ScalePrompt(
                    prompt=[
                        SystemMessage(content=self.system_template.format(
                            persona=persona_text, encouragement=encouragement_text,
                            policy_scale=self.policy_scales[issue_to_scale]
                        )),
                        *example_messages,
                        HumanMessage(content=self.human_template.format(text=text))
                    ], persona=persona_text, encouragement=encouragement_text,
                    persona_idx=persona_idx, encouragement_idx=encouragement_idx
                ))
        return prompts
