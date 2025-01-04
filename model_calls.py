import os
from openai import OpenAI
import time
import logging

logger = logging.getLogger(__name__)


class OpenAIModelCalls:
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

    def __init__(
        self, api_key=None, base_url="https://api.openai.com/v1/", model_name="gpt-4o"
    ):
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

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
