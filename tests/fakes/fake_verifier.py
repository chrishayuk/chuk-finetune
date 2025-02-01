# tests/fakes/fake_verifier.py
class FakeVerifier:
    def check(self, answer: str) -> bool:
        # Just check if the string contains "good".
        return "good" in answer.lower()

def fake_calculate_reward(response: str, verifier) -> float:
    """
    Simple "reward" function: +1 if verifier says good,
    else 0. Or random, etc.
    """
    return 1.0 if verifier.check(response) else 0.0
