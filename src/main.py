import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import faiss
import random

from config import load_config
from data_loader import read_and_split_text
from retriever import DPRRetriever
from generator import GPT2AnswerGenerator


def is_policy_related(question):
    policy_keywords = [
        "policy",
        "vacation",
        "leave",
        "reimbursement",
        "expense",
        "remote",
        "mobile",
        "benefits",
        "salary",
        "sick",
        "holiday",
        "travel",
        "work",
        "employee",
        "hr",
        "office",
        "equipment",
        "company",
        "request",
        "submit",
        "approval",
    ]

    question_lower = question.lower()
    return any(keyword in question_lower for keyword in policy_keywords)


def main():
    config = load_config()

    policy_file = config["data"]["policy_file"]

    context_encoder_name = config["models"]["context_encoder"]
    question_encoder_name = config["models"]["question_encoder"]
    generator_name = config["models"]["generator"]

    top_k = config["retrieval"]["top_k"]
    max_distance = config["retrieval"].get("max_distance", 300)

    max_context_length = config["tokenization"]["max_context_length"]
    max_generation_input_length = config["tokenization"]["max_generation_input_length"]
    max_new_tokens = config["tokenization"]["max_new_tokens"]

    print("Loading company policy document...")
    paragraphs = read_and_split_text(policy_file)

    random.seed(42)
    random.shuffle(paragraphs)

    print("Loading DPR retriever...")
    retriever = DPRRetriever(
        context_encoder_name=context_encoder_name,
        question_encoder_name=question_encoder_name,
        max_context_length=max_context_length,
    )

    print("Encoding policy paragraphs and building FAISS index...")
    retriever.build_index(paragraphs)

    print("Loading GPT-2 generator...")
    generator = GPT2AnswerGenerator(
        model_name=generator_name,
        max_input_length=max_generation_input_length,
        max_new_tokens=max_new_tokens,
    )

    print("\nHR Policy Assistant is ready.")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("Employee question: ").strip()

        if question.lower() in ["exit", "quit", "bye"]:
            print("Goodbye.")
            break

        if not question:
            print("\nGenerated Answer:")
            print("Please enter a valid question.")
            print("-" * 80)
            continue

        if not is_policy_related(question):
            print("\nGenerated Answer:")
            print("I can only answer questions related to company policies.")
            print("-" * 80)
            continue

        contexts, distances = retriever.search(question, top_k=top_k)

        if distances[0] > max_distance:
            print("\nGenerated Answer:")
            print("I could not find relevant policy information for your question.")
            print("-" * 80)
            continue

        answer = generator.generate_answer(question, contexts)

        print("\nRetrieved Contexts:")
        for i, context in enumerate(contexts, start=1):
            print(f"{i}. {context}")

        print("\nRetrieval Distances:")
        print(distances)

        print("\nGenerated Answer:")
        print(answer)
        print("-" * 80)


if __name__ == "__main__":
    main()

