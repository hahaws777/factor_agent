import unittest

from agent.factor_dsl import DSLConfig, compile_expression_to_module, validate_expression


class FactorRecipeLibraryTests(unittest.TestCase):
    def test_all_prepared_recipes_are_valid_dsl(self):
        from agent.factor_recipe_library import PREPARED_FACTOR_RECIPES

        self.assertGreater(len(PREPARED_FACTOR_RECIPES), 0)
        cfg = DSLConfig()
        for recipe in PREPARED_FACTOR_RECIPES:
            with self.subTest(recipe=recipe.recipe_id):
                result = validate_expression(recipe.expression, cfg)
                self.assertTrue(result.is_valid, result.errors)
                code = compile_expression_to_module(recipe.to_spec(), cfg=cfg)
                self.assertIn("FACTOR_EXPRESSION", code)


if __name__ == "__main__":
    unittest.main()
