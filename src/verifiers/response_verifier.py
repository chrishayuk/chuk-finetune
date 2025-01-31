# src/verifiers/response_verifier.py
import re

def check_format(response):
    # checks for a think tag
    has_think = re.search(r'<think>(.*?)</think>', response, re.DOTALL)

    # checks for a answer tag
    has_answer = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)

    # checks for a verifier tag
    has_verifier_answer = re.search(r'<verifier_answer>(.*?)</verifier_answer>', response, re.DOTALL)

    # return a reward
    return 1.0 if (has_think and has_answer and has_verifier_answer) else 0.0

def extract_verifier_answer(response):
    # get the verifier answer
    match = re.search(r'<verifier_answer>(.*?)</verifier_answer>', response, re.DOTALL)

    # return
    return match.group(1).strip() if match else None

def extract_answer(response):
    # extract the answer
    match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)

    # return the answer
    return match.group(1).strip() if match else None

def calculate_reward(response, verifier):
    # calculate the format reward
    format_reward = check_format(response)

    # extract the answer
    answer = extract_answer(response)

    # extract the verifier answer
    answer = extract_verifier_answer(response)

    # accuracy reward
    accuracy_reward = 1.0 if verifier.check(answer) else 0.0

    # return the reward
    return format_reward + accuracy_reward