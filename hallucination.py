#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Title: Mitigating Hallucinations in GPT-2 Small Model with Adjusted Methods
Description: This script modifies the approaches for Fact Verification, Output Calibration, and Prompt Engineering to work more effectively with GPT-2 Small.
Author: Charith Ellepola
Date: [Date]
"""

import re
from typing import Optional, Dict, List

from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_model_and_tokenizer(model_name: str = 'gpt2') -> (GPT2LMHeadModel, GPT2Tokenizer):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()  # Set the model to evaluation mode
    return model, tokenizer


def generate_text(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, prompt: str, max_new_tokens: int = 50,
                  temperature: float = 0.7, do_sample: bool = True) -> str:
    inputs = tokenizer(prompt, return_tensors='pt', padding=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_k=50,
        top_p=0.95,
        num_beams=5,
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return text


def get_author_from_kb(title: str, knowledge_base: Dict[str, str]) -> Optional[str]:
    return knowledge_base.get(title, None)


def generate_grounded_response(prompt: str, knowledge_base: Dict[str, str]) -> str:
    match = re.search(r"'([^']+)'", prompt)
    if match:
        book_title = match.group(1)
        author = get_author_from_kb(book_title, knowledge_base)
        if author:
            response = f"The author of the book '{book_title}' is {author}."
        else:
            response = f"I'm sorry, I don't have information about the book '{book_title}'."
    else:
        response = "Please provide the book title in single quotes."
    return response


def generate_unverified_response(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, prompt: str) -> str:
    # Modify the prompt to guide the model to a consistent format
    prompt += "\n\nPlease answer in the format: 'The author of the book '<Book Title>' is <Author Name>.'"
    response = generate_text(model, tokenizer, prompt, temperature=0.7)
    return response


def verify_response(prompt: str, response: str, knowledge_base: Dict[str, str]) -> str:
    match_prompt = re.search(r"'([^']+)'", prompt)
    if match_prompt:
        book_title = match_prompt.group(1)
    else:
        return "Unable to extract book title from the prompt."

    response_author = extract_author_from_response(response)
    actual_author = get_author_from_kb(book_title, knowledge_base)

    if actual_author is not None:
        # actual_author exists in the knowledge base
        if response_author == actual_author.lower():
            return response  # Correct response
        else:
            return f"The information provided may be incorrect. The actual author of '{book_title}' is {actual_author}."
    else:
        # actual_author is None (book not in knowledge base)
        if response_author is None:
            # Model correctly expressed uncertainty
            return f"I'm sorry, I don't have information about the book '{book_title}'."
        else:
            # Model provided an author for an unknown book (hallucination)
            return f"The information provided may be incorrect. The actual author of '{book_title}' is unknown."


def extract_author_from_response(response: str) -> Optional[str]:
    # Adjusted regex patterns
    patterns = [
        r"The author of the book '.*?' is (.+?)\.",
        r"The author of '.*?' is (.+?)\.",
        r"'.*?' is written by (.+?)\.",
        r"'.*?' was written by (.+?)\.",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).strip().lower()
    # Check for uncertainty expressions
    uncertainty_phrases = [
        "i don't know",
        "i am unsure",
        "i'm sorry, but i don't have information",
        "i do not have information",
        "i'm sorry, i don't know",
        "unable to find",
        "do not know",
        "don't have information",
        "i'm not sure",
        "cannot recall",
        "no information"
    ]
    response_lower = response.lower()
    if any(phrase in response_lower for phrase in uncertainty_phrases):
        return None  # Indicates uncertainty
    else:
        return "hallucinated"  # Indicates a hallucination


def generate_and_verify(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, prompt: str, knowledge_base: Dict[str, str]) -> str:
    unverified_response = generate_unverified_response(model, tokenizer, prompt)
    verified_response = verify_response(prompt, unverified_response, knowledge_base)
    return verified_response


def generate_calibrated_response(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, prompt: str) -> str:
    # Provide examples in the prompt
    calibration_prompt = (
        "Answer the following questions accurately. If you are unsure or don't know the answer, please say so.\n\n"
        "Q: Who is the author of the book 'Unknown Book'?\n"
        "A: I'm sorry, I don't have information about the book 'Unknown Book'.\n\n"
        f"Q: {prompt}\nA:"
    )
    response = generate_text(model, tokenizer, calibration_prompt, temperature=0.7)
    return response.strip()


def generate_engineered_response(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, prompt: str) -> str:
    # Use few-shot learning with examples
    engineered_prompt = (
        "As a knowledgeable assistant, please provide accurate information based on known data. "
        "If you don't know the answer, express uncertainty.\n\n"
        "Q: Who is the author of the book 'Unknown Book'?\n"
        "A: I'm sorry, I don't have information about the book 'Unknown Book'.\n\n"
        "Q: Who is the author of the book '1984'?\n"
        "A: The author of the book '1984' is George Orwell.\n\n"
        f"Q: {prompt}\nA:"
    )
    response = generate_text(model, tokenizer, engineered_prompt, temperature=0.7)
    return response.strip()


def evaluate_responses(responses: List[str], test_data: List[Dict]) -> Dict[str, float]:
    correct = 0
    hallucinations = 0
    uncertainties = 0
    total = len(test_data)

    for response, data in zip(responses, test_data):
        ground_truth = data['answer']
        prompt = data['prompt']

        response_author = extract_author_from_response(response)
        ground_truth_author = ground_truth.lower() if ground_truth else None

        if response_author == ground_truth_author:
            correct += 1
        elif response_author == "hallucinated":
            hallucinations += 1
        elif response_author is None:
            if ground_truth_author is None:
                uncertainties += 1  # Correctly expressed uncertainty
            else:
                hallucinations += 1  # Failed to provide known author
        else:
            hallucinations += 1  # Incorrect author provided

    accuracy = (correct / total) * 100
    hallucination_rate = (hallucinations / total) * 100
    uncertainty_rate = (uncertainties / total) * 100

    return {
        "Accuracy": accuracy,
        "Hallucination Rate": hallucination_rate,
        "Uncertainty Rate": uncertainty_rate
    }


def main():
    model, tokenizer = load_model_and_tokenizer()

    knowledge_base = {
        "The Lost City of Z": "David Grann",
        "1984": "George Orwell",
        "To Kill a Mockingbird": "Harper Lee",
        "The Great Gatsby": "F. Scott Fitzgerald",
        "Moby-Dick": "Herman Melville"
    }

    test_data = [
        {
            "prompt": "Who is the author of the book 'The Lost City of Z'?",
            "answer": "David Grann"
        },
        {
            "prompt": "Who is the author of the book '1984'?",
            "answer": "George Orwell"
        },
        {
            "prompt": "Who is the author of the book 'Unknown Book'?",
            "answer": None  # Indicating that the book is unknown
        }
    ]

    technique_responses = {
        "Baseline": [],
        "Grounded Generation": [],
        "Fact Verification": [],
        "Output Calibration": [],
        "Prompt Engineering": []
    }

    for data in test_data:
        prompt = data['prompt']

        # Baseline
        response = generate_text(model, tokenizer, prompt, temperature=0.7)
        technique_responses["Baseline"].append(response)

        # Grounded Generation
        response = generate_grounded_response(prompt, knowledge_base)
        technique_responses["Grounded Generation"].append(response)

        # Fact Verification
        response = generate_and_verify(model, tokenizer, prompt, knowledge_base)
        technique_responses["Fact Verification"].append(response)

        # Output Calibration
        response = generate_calibrated_response(model, tokenizer, prompt)
        technique_responses["Output Calibration"].append(response)

        # Prompt Engineering
        response = generate_engineered_response(model, tokenizer, prompt)
        technique_responses["Prompt Engineering"].append(response)

    for technique, responses in technique_responses.items():
        print(f"\n--- {technique} Responses ---")
        for response, data in zip(responses, test_data):
            print(f"Prompt: {data['prompt']}\nResponse: {response}\n")

        metrics = evaluate_responses(responses, test_data)
        print(f"Metrics for {technique}:")
        print(f"Accuracy: {metrics['Accuracy']:.2f}%")
        print(f"Hallucination Rate: {metrics['Hallucination Rate']:.2f}%")
        print(f"Uncertainty Rate: {metrics['Uncertainty Rate']:.2f}%\n")


if __name__ == "__main__":
    main()
