import ast
import unittest

from agent.factor_dsl import DSLConfig, FactorSpec, compile_expression_to_module, validate_expression


class FactorDSLTests(unittest.TestCase):
    def test_valid_expression_compiles_to_python(self):
        spec = FactorSpec(
            name="ranked_momentum",
            family="momentum",
            economic_hypothesis="Stocks with stronger trailing returns continue to outperform.",
            expression="rank(ts_return(close, 20))",
            expected_sign="positive",
        )
        result = validate_expression(spec.expression)
        self.assertTrue(result.is_valid, result.errors)
        code = compile_expression_to_module(spec)
        ast.parse(code)
        self.assertIn("def compute_factor_df", code)

    def test_rejects_unknown_field(self):
        result = validate_expression("rank(next_return)", DSLConfig())
        self.assertFalse(result.is_valid)

    def test_rejects_large_window(self):
        result = validate_expression("ts_mean(close, 999)", DSLConfig(max_window=252))
        self.assertFalse(result.is_valid)

    def test_rejects_attribute_access(self):
        result = validate_expression("__import__('os').system('echo bad')")
        self.assertFalse(result.is_valid)


if __name__ == "__main__":
    unittest.main()

