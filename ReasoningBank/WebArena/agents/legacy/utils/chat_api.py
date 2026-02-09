# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import asdict, dataclass
import io
import json
from .prompt_templates import PromptTemplate, get_prompt_template
from langchain.schema import BaseMessage, SystemMessage, HumanMessage, AIMessage
from functools import partial
from typing import Optional, List, Any
import logging
from typing import Tuple
import time
from google import genai
from google.genai.types import HttpOptions, GenerateContentConfig
from typing import Callable
from pydantic import PrivateAttr
from langchain_community.llms import HuggingFaceHub, HuggingFacePipeline
# from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage
from langchain.chat_models.base import SimpleChatModel
from langchain_anthropic import ChatAnthropic
from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field
from transformers import pipeline
from dataclasses import dataclass
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer
from transformers import GPT2TokenizerFast
from openai import OpenAI


@dataclass
class ChatModelArgs:
    """Serializable object for instantiating a generic chat model.

    Attributes
    ----------
    model_name : str
        The name or path of the model to use.
    model_url : str, optional
        The url of the model to use, e.g. via TGI. If None, then model_name or model_path must
        be specified.
    eai_token: str, optional
        The EAI token to use for authentication on Toolkit. Defaults to snow.optimass_account.cl4code's token.
    temperature : float
        The temperature to use for the model.
    max_new_tokens : int
        The maximum number of tokens to generate.
    hf_hosted : bool
        Whether the model is hosted on HuggingFace Hub. Defaults to False.
    info : dict, optional
        Any other information about how the model was finetuned.
    DGX related args
    n_gpus : int
        The number of GPUs to use. Defaults to 1.
    tgi_image : str
        The TGI image to use. Defaults to "e3cbr6awpnoq/research/text-generation-inference:1.1.0".
    ace : str
        The ACE to use. Defaults to "servicenow-scus-ace".
    workspace : str
        The workspace to use. Defaults to UI_COPILOT_SCUS_WORKSPACE.
    max_total_tokens : int
        The maximum number of total tokens (input + output). Defaults to 4096.
    """

    model_name: str = "openai/gpt-3.5-turbo"
    model_url: str = None
    temperature: float = 0.0
    max_new_tokens: int = None
    max_total_tokens: int = None
    max_input_tokens: int = None
    hf_hosted: bool = False
    info: dict = None
    n_retry_server: int = 4

    def __post_init__(self):
        if self.model_url is not None and self.hf_hosted:
            raise ValueError("model_url cannot be specified when hf_hosted is True")

    def make_chat_model(self):
        if self.model_name.startswith("openai"):
            _, model_name = self.model_name.split("/")
            return ChatOpenAI(
                model_name=model_name,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
            )
        elif self.model_name.startswith("gemini"):
            return ChatGemini(
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
            )
        elif self.model_name.startswith("claude"):
            return ChatClaude(
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
            )

        else:
            return HuggingFaceChatModel(
                model_name=self.model_name,
                hf_hosted=self.hf_hosted,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                max_total_tokens=self.max_total_tokens,
                max_input_tokens=self.max_input_tokens,
                model_url=self.model_url,
                eai_token=None,
                n_retry_server=self.n_retry_server,
            )

    @property
    def model_short_name(self):
        if "/" in self.model_name:
            return self.model_name.split("/")[1]
        else:
            return self.model_name

    def key(self):
        """Return a unique key for these arguments."""
        return json.dumps(asdict(self), sort_keys=True)

    def has_vision(self):
        # TODO make sure to upgrade this as we add more models
        name_patterns_with_vision = [
            "vision",
            "4o",
        ]
        return any(pattern in self.model_name for pattern in name_patterns_with_vision)

class ChatClaude(SimpleChatModel):
    """
    Custom LLM Chatbot that can interface with Anthropic Claude models.

    This class allows for the creation of a custom chatbot using Anthropic Claude models.
    It provides flexibility in defining the temperature for response sampling and the maximum number of new tokens
    in the response.

    Attributes:
        model_name (str): The name of the Claude model to use.
        temperature (float): The temperature for response sampling.
        max_new_tokens (int): The maximum number of new tokens in the response.
    """

    _llm: Callable = PrivateAttr()
    n_retry_server: int = Field(default=4)

    def __init__(
        self,
        model_name,
        temperature,
        max_tokens,
        n_retry_server=4,
    ):
        """
        Initializes the CustomLLMChatbot with the specified configurations.

        Args:
            model_name (str): The path to the model checkpoint.
            prompt_template (PromptTemplate, optional): A string template for structuring the prompt.
            hf_hosted (bool, optional): Whether the model is hosted on HuggingFace Hub. Defaults to False.
            temperature (float, optional): Sampling temperature. Defaults to 0.1.
            max_new_tokens (int, optional): Maximum length for the response. Defaults to 64.
            model_url (str, optional): The url of the model to use. If None, then model_name or model_name will be used. Defaults to None.
        """
        logging.info("Loading Claude")
        import os
        from anthropic import AnthropicVertex
        client = AnthropicVertex(region=os.environ.get("GOOGLE_CLOUD_LOCATION"), project_id=os.environ.get("GOOGLE_CLOUD_PROJECT"))

        llm_fn = partial(
            client.messages.create,
            model=model_name,
            temperature=temperature,
            max_tokens=4096,
        )
        super().__init__()
        self._llm = llm_fn
        self.n_retry_server = n_retry_server
        # self.n_retry_server = n_retry_server



    def _call(
        self,
        messages,
        stop = None,
        run_manager = None,
        **kwargs,
    ):
        if stop is not None or run_manager is not None or kwargs:
            logging.warning(
                "The `stop`, `run_manager`, and `kwargs` arguments are ignored in this implementation."
            )

        # if self.tokenizer:

            # prompt = self.tokenizer.apply_chat_template(messages_formated, tokenize=False)

        # if self.prompt_template:
        #     prompt = self.prompt_template.construct_prompt(messages)
        # else:

        raw = _convert_messages_to_dict(messages)
        # prompt = "\n\n".join(m["content"] for m in messages_formated) + "\n\n"

        system_blocks = []
        new_messages = []
        for m in raw:
            role = m.get("role")
            content = m.get("content", "")
            if role == "system":
                # Anthropic 顶层 system 可以是 str 或 content blocks；这里简单拼成 str
                system_blocks.append(content if isinstance(content, str) else str(content))
            elif role in ("user", "assistant"):
                new_messages.append(m)
            else:
                # 其它角色（如 tool）视需要处理；最简单做法：跳过或映射
                new_messages.append(m)

        system_text = "\n".join(system_blocks) if system_blocks else None

        itr = 0
        while True:
            try:
                response = self._llm(system=system_text, messages=new_messages).content[0].text
                return response
            except Exception as e:
                if itr == self.n_retry_server - 1:
                    raise e
                logging.warning(
                    f"Failed to get a response from the server: \n{e}\n"
                    f"Retrying... ({itr+1}/{self.n_retry_server})"
                )
                time.sleep(5)
                itr += 1

    def _llm_type(self):
        return "claude"


class ChatGemini(SimpleChatModel):
    """
    Custom LLM Chatbot that can interface with Google Gemini models.

    This class allows for the creation of a custom chatbot using Google Gemini models.
    It provides flexibility in defining the temperature for response sampling and the maximum number of new tokens
    in the response.

    Attributes:
        model_name (str): The name of the Gemini model to use.
        temperature (float): The temperature for response sampling.
        max_new_tokens (int): The maximum number of new tokens in the response.
    """

    _llm: Callable = PrivateAttr()
    n_retry_server: int = Field(default=4)

    def __init__(
        self,
        model_name,
        temperature,
        max_tokens,
        n_retry_server=4,
    ):
        """
        Initializes the CustomLLMChatbot with the specified configurations.

        Args:
            model_name (str): The path to the model checkpoint.
            prompt_template (PromptTemplate, optional): A string template for structuring the prompt.
            hf_hosted (bool, optional): Whether the model is hosted on HuggingFace Hub. Defaults to False.
            temperature (float, optional): Sampling temperature. Defaults to 0.1.
            max_new_tokens (int, optional): Maximum length for the response. Defaults to 64.
            model_url (str, optional): The url of the model to use. If None, then model_name or model_name will be used. Defaults to None.
        """
        logging.info("Loading Gemini")
        client = genai.Client(http_options=HttpOptions(api_version="v1"))

        llm_fn = partial(
            client.models.generate_content,
            model=model_name,
            contents="",
            config=GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        super().__init__()
        self._llm = llm_fn
        self.n_retry_server = n_retry_server


    def _call(
        self,
        messages,
        stop = None,
        run_manager = None,
        **kwargs,
    ):
        if stop is not None or run_manager is not None or kwargs:
            logging.warning(
                "The `stop`, `run_manager`, and `kwargs` arguments are ignored in this implementation."
            )

        # if self.tokenizer:

            # prompt = self.tokenizer.apply_chat_template(messages_formated, tokenize=False)

        # if self.prompt_template:
        #     prompt = self.prompt_template.construct_prompt(messages)
        # else:

        messages_formated = _convert_messages_to_dict(messages)
        prompt = "\n\n".join(m["content"] for m in messages_formated) + "\n\n"

        itr = 0
        while True:
            try:
                response = self._llm(contents=prompt).text
                return response
            except Exception as e:
                if itr == self.n_retry_server - 1:
                    raise e
                logging.warning(
                    f"Failed to get a response from the server: \n{e}\n"
                    f"Retrying... ({itr+1}/{self.n_retry_server})"
                )
                time.sleep(5)
                itr += 1

    def _llm_type(self):
        return "gemini"


class HuggingFaceChatModel(SimpleChatModel):
    """
    Custom LLM Chatbot that can interface with HuggingFace models.

    This class allows for the creation of a custom chatbot using models hosted
    on HuggingFace Hub or a local checkpoint. It provides flexibility in defining
    the temperature for response sampling and the maximum number of new tokens
    in the response.

    Attributes:
        llm (Any): The HuggingFaceHub model instance.
        prompt_template (Any): Template for the prompt to be used for the model's input sequence.
    """

    llm: Any = Field(description="The HuggingFaceHub model instance")
    # tokenizer: Any = Field(
    #     default=None,
    #     description="The tokenizer to use for the model",
    # )
    # prompt_template: Optional[PromptTemplate] = Field(
    #     default=None,
    #     description="Template for the prompt to be used for the model's input sequence",
    # )
    n_retry_server: int = Field(
        default=4,
        description="The number of times to retry the server if it fails to respond",
    )
    model_name: str = Field(
        default="",
        description="The name of the model",
    )

    def __init__(
        self,
        model_name,
        hf_hosted,
        temperature,
        max_new_tokens,
        max_total_tokens,
        max_input_tokens,
        model_url,
        eai_token,
        n_retry_server,
    ):
        """
        Initializes the CustomLLMChatbot with the specified configurations.

        Args:
            model_name (str): The path to the model checkpoint.
            prompt_template (PromptTemplate, optional): A string template for structuring the prompt.
            hf_hosted (bool, optional): Whether the model is hosted on HuggingFace Hub. Defaults to False.
            temperature (float, optional): Sampling temperature. Defaults to 0.1.
            max_new_tokens (int, optional): Maximum length for the response. Defaults to 64.
            model_url (str, optional): The url of the model to use. If None, then model_name or model_name will be used. Defaults to None.
        """

        if max_new_tokens is None:
            max_new_tokens = max_total_tokens - max_input_tokens
            logging.warning(
                f"max_new_tokens is not specified. Setting it to {max_new_tokens} (max_total_tokens - max_input_tokens)."
            )

        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # if isinstance(self.tokenizer, GPT2TokenizerFast):
        #     # TODO: make this less hacky once tokenizer.apply_chat_template is more mature
        #     logging.warning(
        #         f"No chat template is defined for {model_name}. Resolving to the hard-coded templates."
        #     )
        #     self.tokenizer = None
        #     self.prompt_template = get_prompt_template(model_name)

        if temperature < 1e-3:
            logging.warning(
                "some weird things might happen when temperature is too low for some models."
            )

        model_kwargs = {
            "temperature": temperature,
        }

        if model_url is not None:
            logging.info("Loading the LLM from a URL")
            client = InferenceClient(model=model_url, token=eai_token)
            llm_fn = partial(
                client.text_generation, temperature=temperature, max_new_tokens=max_new_tokens
            )
        elif hf_hosted:
            logging.info("Serving the LLM on HuggingFace Hub")
            model_kwargs["max_length"] = max_new_tokens
            llm_fn = HuggingFaceHub(repo_id=model_name, model_kwargs=model_kwargs)
        else:
            logging.info("Loading the LLM locally")

            # pipe = pipeline(
            #     task="text-generation",
            #     model=model_name,
            #     device_map="auto",
            #     max_new_tokens=max_new_tokens,
            #     model_kwargs=model_kwargs,
            #     return_full_text=False
            # )
            # llm_fn = HuggingFacePipeline(pipeline=pipe)

            # llm_fn = LLM(model=model_name)

            llm_fn = OpenAI(
                base_url="http://localhost:8000/v1",
                api_key="token-abc123",
            )

        super().__init__(
            llm = llm_fn,
            n_retry_server = n_retry_server,
            model_name = model_name,
            )

    def _call(
        self,
        messages,
        stop = None,
        run_manager = None,
        **kwargs,
    ):
        if stop is not None or run_manager is not None or kwargs:
            logging.warning(
                "The `stop`, `run_manager`, and `kwargs` arguments are ignored in this implementation."
            )

        # if self.tokenizer:
        #     messages_formated = _convert_messages_to_dict(messages)
        #     prompt = self.tokenizer.apply_chat_template(messages_formated, tokenize=False)

        # elif self.prompt_template:
        #     prompt = self.prompt_template.construct_prompt(messages)
        messages_formated = _convert_messages_to_dict(messages)
        prompt = "\n\n".join(m["content"] for m in messages_formated) + "\n\n"

        # params = SamplingParams(
        #     temperature=kwargs.get("temperature", 0.0),
        #     max_tokens=kwargs.get("max_new_tokens", 16384),
        # )

        itr = 0
        while True:
            try:
                # response = self.llm(prompt)
                # response = self.llm.generate(prompt, sampling_params=params)
                # response = response[0].outputs[0].text
                response = self.llm.chat.completions.create(
                    model=self.model_name,
                    messages=messages_formated,
                )
                response = response.choices[0].message.content
                return response
            except Exception as e:
                if itr == self.n_retry_server - 1:
                    raise e
                logging.warning(
                    f"Failed to get a response from the server: \n{e}\n"
                    f"Retrying... ({itr+1}/{self.n_retry_server})"
                )
                time.sleep(5)
                itr += 1

    def _llm_type(self):
        return "huggingface"


def _convert_messages_to_dict(messages):
    """
    Converts a list of message objects into a list of dictionaries, categorizing each message by its role.

    Each message is expected to be an instance of one of the following types: SystemMessage, HumanMessage, AIMessage.
    The function maps each message to its corresponding role ('system', 'user', 'assistant') and formats it into a dictionary.

    Args:
        messages (list): A list of message objects.

    Returns:
        list: A list of dictionaries where each dictionary represents a message and contains 'role' and 'content' keys.

    Raises:
        ValueError: If an unsupported message type is encountered.

    Example:
        >>> messages = [SystemMessage("System initializing..."), HumanMessage("Hello!"), AIMessage("How can I assist?")]
        >>> _convert_messages_to_dict(messages)
        [
            {"role": "system", "content": "System initializing..."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "How can I assist?"}
        ]
    """

    # Mapping of message types to roles
    message_type_to_role = {
        SystemMessage: "system",
        HumanMessage: "user",
        AIMessage: "assistant",
    }

    chat = []
    for message in messages:
        message_role = message_type_to_role.get(type(message))
        if message_role:
            chat.append({"role": message_role, "content": message.content})
        else:
            raise ValueError(f"Message type {type(message)} not supported")

    return chat
