import json
import logging

# custom imports
from mcts_tree import MCTS
from node import QuestionNode
from model_calls import OpenAIModelCalls


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

num_iterations = 4
qa_pair = json.load(open("qa.json", "r"))
mcts = MCTS()
model_api = OpenAIModelCalls(base_url="https://www.dmxapi.com/v1", model_name="gpt-4o")
root_node = QuestionNode(qa_pair["question"], model_api, None)

for _ in range(num_iterations):
    mcts.do_iteration(root_node, qa_pair["ground_truth"])

best_next_node = mcts.choose(root_node)
logging.info(f"Best next node: ")
logging.info(
    best_next_node.previous_state + best_next_node.prompt + best_next_node.output
)
