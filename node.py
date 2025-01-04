from abc import ABC, abstractmethod
from collections import namedtuple
import random
import re
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def find_children(self):
        "All possible successors of this board state"
        return set()

    @abstractmethod
    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        return None

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        return True

    @abstractmethod
    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        return 0


class LLMNode(Node):
    prompt = "LLM: "

    def __init__(self, previous_state, model_api, parent=None):
        self.previous_state = previous_state
        self.parent = parent
        self.model_api = model_api
        self.state = self.make_a_move(previous_state)
        self.children = None

    def find_children(self):
        if not self.children:
            # if there are 3 llms, silly llm, funny llm, smart llm
            self.children = [
                SillyNode(self.state, self.model_api, self),
                FunnyNode(self.state, self.model_api, self),
                SmartNode(self.state, self.model_api, self),
            ]
        return self.children

    def find_random_child(self):
        return random.choice([SillyNode, FunnyNode, SmartNode])(
            self.state, self.model_api, self
        )

    def is_terminal(self):
        text = self.output
        pattern = r"final answer:\s*(-?\d+(\.\d+)?)"
        match = re.search(pattern, text, re.IGNORECASE)
        return match

    def reward(self, ground_truth):
        assert self.is_terminal(), f"reward called on non-terminal node {self}"
        match = re.search(
            r"final answer:\s*(-?\d+(\.\d+)?)", self.output, re.IGNORECASE
        )
        prediction = float(match.group(1))
        logger.info(f"Solving path: {self.previous_state + self.prompt + self.state}")
        if prediction == ground_truth:
            return 1
        return 0

    def make_a_move(
        self,
        previous_state,
    ):
        self.output = self.model_api.get_output(previous_state + self.prompt)
        return previous_state + self.prompt + self.output


class QuestionNode(LLMNode):
    def make_a_move(self, previous_state):
        # for question node, the current state is the question
        self.output = previous_state
        return previous_state


class SillyNode(LLMNode):
    prompt = "Silly Man: "


class FunnyNode(LLMNode):
    prompt = "Funny Man: "


class SmartNode(LLMNode):
    prompt = "Smart Man: "
