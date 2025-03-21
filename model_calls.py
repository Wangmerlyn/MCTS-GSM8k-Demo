import os
from openai import OpenAI
import time
import logging

logger = logging.getLogger(__name__)


class ModelCalls:
    """Base model calls class, serving as the parent class for all model implementations"""
    system_prompt = """You are participating in a collaborative problem-solving conversation with three distinct personas:

A smart person, who is logical and focused on finding the best solution.
A witty, humorous, and romantic person, who approaches the problem with creativity and charm.
A silly but lovable person, who may appear naive but adds a unique perspective.
The group solves problems by breaking them down into step-by-step tasks, with each person contributing to one step at a time.

You are tasked with role-playing as one of these personas and contributing only your character's response for one step. Here are the guidelines:

Do not respond as the other personas—only speak as the persona assigned for this turn.
Always focus on solving just one step of the problem. Never think or respond beyond a single step.
If, at any point, the information gathered so far is sufficient to directly solve the problem, output the final solution in this format:
"Final answer: [a number]"
Do not include any additional commentary or explanation.
Otherwise, continue contributing to only one step of the solution process based on your persona's unique traits.
"""

    def __init__(self, api_key=None, base_url=None, model_name=None):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.client = None
        # Initialize token usage tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.api_calls = 0

    def get_output(self, prompt, max_retries=3, retry_delay=2):
        """
        Abstract method to get model output
        
        Args:
            prompt (str): The user input prompt
            max_retries (int): Maximum number of retry attempts. Default is 3
            retry_delay (int): Delay between retries in seconds. Default is 2
            
        Returns:
            str: The model response
        """
        raise NotImplementedError("Subclasses must implement get_output method")
    
    def get_token_usage(self):
        """
        Get token usage statistics
        
        Returns:
            dict: Dictionary containing token usage statistics
        """
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "api_calls": self.api_calls
        }
    
    def add_token_usage(self, prompt_tokens, completion_tokens):
        """
        Add token usage from an API call
        
        Args:
            prompt_tokens (int): Number of tokens in the prompt
            completion_tokens (int): Number of tokens in the completion
        """
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        self.api_calls += 1


class OpenAIModelCalls(ModelCalls):
    """OpenAI API call implementation"""
    
    def __init__(
        self, api_key=None, base_url="https://api.openai.com/v1/", model_name="gpt-4o"
    ):
        super().__init__(api_key, base_url, model_name)
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def get_output(self, prompt, max_retries=3, retry_delay=2):
        """
        Get the output from the chat completion API with retry logic.

        Args:
            prompt (str): The user input prompt.
            max_retries (int): Maximum number of retry attempts. Default is 3.
            retry_delay (int): Delay between retries in seconds. Default is 2.

        Returns:
            str: The chat completion response.
        """
        attempt = 0
        while attempt < max_retries:
            try:
                # Call the API
                chat_completion = self.client.chat.completions.create(
                    temperature=0,
                    messages=[
                        {
                            "role": "system",
                            "content": [
                                {
                                    "type": "text",
                                    "text": self.system_prompt,
                                },
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt,
                                },
                            ],
                        },
                    ],
                    model=self.model_name,
                )
                
                # Track token usage
                if hasattr(chat_completion, 'usage') and chat_completion.usage:
                    prompt_tokens = chat_completion.usage.prompt_tokens
                    completion_tokens = chat_completion.usage.completion_tokens
                    self.add_token_usage(prompt_tokens, completion_tokens)
                    logger.debug(f"API call tokens: {prompt_tokens} prompt, {completion_tokens} completion")
                
                # Extract the message from the API response
                message = chat_completion.choices[0].message.content
                return message

            except Exception as e:
                # Log the error and retry
                attempt += 1
                logger.error(f"Attempt {attempt} failed with error: {e}")
                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error("Max retries reached. Raising exception.")
                    raise


class DeepSeekModelCalls(ModelCalls):
    """DeepSeek API call implementation"""
    
    def __init__(
        self, api_key=None, base_url="https://api.deepseek.com/v1", model_name="deepseek-chat"
    ):
        super().__init__(api_key, base_url, model_name)
        if api_key is None:
            api_key = os.getenv("DEEPSEEK_API_KEY")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def get_output(self, prompt, max_retries=3, retry_delay=2):
        """
        Get output from DeepSeek API with retry logic
        
        Args:
            prompt (str): The user input prompt
            max_retries (int): Maximum number of retry attempts. Default is 3
            retry_delay (int): Delay between retries in seconds. Default is 2
            
        Returns:
            str: The chat completion response
        """
        attempt = 0
        while attempt < max_retries:
            try:
                # Call the API
                chat_completion = self.client.chat.completions.create(
                    temperature=0,
                    messages=[
                        {
                            "role": "system",
                            "content": self.system_prompt,
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    model=self.model_name,
                )
                
                # Track token usage
                if hasattr(chat_completion, 'usage') and chat_completion.usage:
                    prompt_tokens = chat_completion.usage.prompt_tokens
                    completion_tokens = chat_completion.usage.completion_tokens
                    self.add_token_usage(prompt_tokens, completion_tokens)
                    logger.debug(f"API call tokens: {prompt_tokens} prompt, {completion_tokens} completion")
                
                # Extract the message from the API response
                message = chat_completion.choices[0].message.content
                return message

            except Exception as e:
                # Log the error and retry
                attempt += 1
                logger.error(f"Attempt {attempt} failed with error: {e}")
                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error("Max retries reached. Raising exception.")
                    raise


# Factory function to create appropriate model calls object
def create_model_calls(provider="openai", api_key=None, base_url=None, model_name=None):
    """
    Create appropriate model calls object
    
    Args:
        provider (str): The model provider, options: "openai", "deepseek"
        api_key (str): API key, will read from environment variables if None
        base_url (str): API base URL
        model_name (str): Model name
        
    Returns:
        ModelCalls: The appropriate model calls object
    """
    provider = provider.lower()
    
    if provider == "openai":
        return OpenAIModelCalls(
            api_key=api_key, 
            base_url=base_url or "https://api.openai.com/v1/",
            model_name=model_name or "gpt-4o"
        )
    elif provider == "deepseek":
        return DeepSeekModelCalls(
            api_key=api_key,
            base_url=base_url or "https://api.deepseek.com/v1",
            model_name=model_name or "deepseek-chat"
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")
