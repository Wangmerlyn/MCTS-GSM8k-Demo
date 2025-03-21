import json
import logging
import argparse

# custom imports
from mcts_tree import MCTS
from node import QuestionNode
from model_calls import create_model_calls


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

def main():
    """Main function to handle command line arguments and execute MCTS"""
    parser = argparse.ArgumentParser(description="Using MCTS to solve GSM8K problems")
    parser.add_argument("--provider", type=str, default="openai", 
                        choices=["openai", "deepseek"],
                        help="Model provider (default: openai)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (default depends on provider)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key (default: read from environment variable)")
    parser.add_argument("--base-url", type=str, default=None,
                        help="API base URL (default depends on provider)")
    parser.add_argument("--iterations", type=int, default=4,
                        help="Number of MCTS iterations (default: 4)")
    args = parser.parse_args()

    num_iterations = args.iterations
    qa_pair = json.load(open("qa.json", "r"))
    
    # Create model calls object using factory function
    model_api = create_model_calls(
        provider=args.provider,
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model
    )
    
    logger.info(f"Using model provider: {args.provider}")
    logger.info(f"Model name: {model_api.model_name}")
    
    # Initialize MCTS and root node
    mcts = MCTS()
    root_node = QuestionNode(qa_pair["question"], model_api, None)

    # Perform MCTS iterations
    for i in range(num_iterations):
        logger.info(f"Performing MCTS iteration {i+1}/{num_iterations}")
        mcts.do_iteration(root_node, qa_pair["ground_truth"])

    # Choose best node
    best_next_node = mcts.choose(root_node)
    logger.info(f"Best next node: ")
    logger.info(
        best_next_node.previous_state + best_next_node.prompt + best_next_node.output
    )
    
    # Display token usage statistics
    token_usage = model_api.get_token_usage()
    logger.info("=" * 40)
    logger.info("Token Usage Statistics:")
    logger.info(f"Total API calls: {token_usage['api_calls']}")
    logger.info(f"Prompt tokens: {token_usage['prompt_tokens']}")
    logger.info(f"Completion tokens: {token_usage['completion_tokens']}")
    logger.info(f"Total tokens: {token_usage['total_tokens']}")
    
    # If we know the cost per token, we could also calculate the approximate cost
    if args.provider == "openai" and model_api.model_name in ["gpt-4o", "gpt-4"]:
        # Approximate costs for GPT-4 models (in USD per 1K tokens)
        prompt_cost = 0.01  # $0.01 per 1K tokens for input
        completion_cost = 0.03  # $0.03 per 1K tokens for output
        total_cost = (token_usage['prompt_tokens'] * prompt_cost / 1000) + (token_usage['completion_tokens'] * completion_cost / 1000)
        logger.info(f"Estimated cost: ${total_cost:.4f} USD")
    logger.info("=" * 40)

if __name__ == "__main__":
    main()
