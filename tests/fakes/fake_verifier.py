# tests/fakes/fake_verifier.py
class FakeVerifier:
    def check(self, answer: str) -> bool:
        # Just check if the string contains "good".
        return "good" in answer.lower()

def fake_calculate_reward(response: str, item: dict):
    """
    Expects item like {"prompt": "...", "verifiers": [...]}
    Returns (score, feedback_text).
    """
    # If the item has a non-empty "verifiers" list, give +1
    if item.get("verifiers"):
        return (1.0, "Found verifiers => +1")
    else:
        return (0.0, "No verifiers => +0")


