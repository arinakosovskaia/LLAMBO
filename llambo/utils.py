import os
from openai import AsyncOpenAI, OpenAI
from transformers import AutoTokenizer
import tiktoken

def initialize_client(model_name):
    """
    Initializes OpenAI client based on the model name.

    :param model_name:
    :return: initialized client.
    """
    if model_name.lower().startswith("gpt"):
        # Настройка для GPT моделей
        client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    else:
        client = AsyncOpenAI(
            base_url="https://api.studio.nebius.ai/v1/",
            api_key=os.environ.get("NEBIUS_API_KEY"),
        )
    return client

model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
llama_tokenizer = AutoTokenizer.from_pretrained(model_id)

def count_tokens(model, text):
    """
    Returns the number of tokens for the given text based on the specified model.

    :param model: Model name (str). If it starts with 'gpt', uses Tiktoken tokenizer.
                  Otherwise, uses the LLaMA tokenizer.
    :param text: Input text (str).
    :return: Number of tokens (int).
    """
    if model.lower().startswith("gpt"):
        # Use Tiktoken tokenizer for GPT models
        encoding = tiktoken.encoding_for_model(model)
        num_tokens = len(encoding.encode(text))
    else:
        # Use LLaMA tokenizer for non-GPT models
        tokens = llama_tokenizer.encode(text, add_special_tokens=True)
        num_tokens = len(tokens)
    
    return num_tokens
