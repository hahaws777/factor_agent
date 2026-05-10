import unittest

from agent.factor_safety import validate_factor_code


class FactorSafetyTests(unittest.TestCase):
    def test_rejects_negative_shift(self):
        code = """
import pandas as pd
def compute_factor_df() -> pd.DataFrame:
    df = pd.DataFrame({"x": [1, 2, 3]})
    return df.shift(-1)
"""
        report = validate_factor_code(code)
        self.assertFalse(report.is_safe)
        self.assertEqual(report.severity, "REJECT")

    def test_rejects_external_api_and_exec(self):
        code = """
import requests
def compute_factor_df():
    exec("print(1)")
    return requests.get("https://example.com")
"""
        report = validate_factor_code(code)
        self.assertFalse(report.is_safe)
        self.assertTrue(any("dangerous" in reason or "external" in reason for reason in report.reasons))

    def test_allows_trailing_rolling_dataframe(self):
        code = """
import pandas as pd
def compute_factor_df() -> pd.DataFrame:
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    out = df.rolling(2).mean()
    return out
"""
        report = validate_factor_code(code)
        self.assertTrue(report.is_safe)
        self.assertIn(report.severity, {"PASS", "WARN"})

    def test_ignores_suspicious_words_in_docstrings(self):
        code = '''
import pandas as pd
def compute_factor_df() -> pd.DataFrame:
    """Hypothesis may mention future returns in prose, not code."""
    return pd.DataFrame({"x": [1.0, 2.0]})
'''
        report = validate_factor_code(code)
        self.assertTrue(report.is_safe)

    def test_rejects_obvious_non_dataframe_return(self):
        code = """
def compute_factor_df():
    return [1, 2, 3]
"""
        report = validate_factor_code(code)
        self.assertFalse(report.is_safe)

    def test_rejects_label_name_y(self):
        code = """
import pandas as pd
def compute_factor_df() -> pd.DataFrame:
    y = pd.DataFrame({"x": [1.0]})
    return y
"""
        report = validate_factor_code(code)
        self.assertFalse(report.is_safe)


if __name__ == "__main__":
    unittest.main()
