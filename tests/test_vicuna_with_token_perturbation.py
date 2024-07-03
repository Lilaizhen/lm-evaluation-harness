import unittest
from lm_eval.models.vicuna_with_token_perturbation import VicunaWithTokenPerturbation
from lm_eval.api.instance import Instance

class TestVicunaWithTokenPerturbation(unittest.TestCase):
    def setUp(self):
        self.model = VicunaWithTokenPerturbation(perturbation_level=0.1)

    def test_loglikelihood(self):
        requests = [
            Instance(
                request_type="loglikelihood",
                doc={"text": "some doc text"},
                arguments=("input text", "target text"),
                idx=0
            )
        ]
        results = self.model.loglikelihood(requests)
        self.assertEqual(len(results), 1)
        # 更多断言以验证结果

    def test_loglikelihood_rolling(self):
        requests = [
            Instance(
                request_type="loglikelihood_rolling",
                doc={"text": "some doc text"},
                arguments=("input text",),
                idx=0
            )
        ]
        results = self.model.loglikelihood_rolling(requests)
        self.assertEqual(len(results), 1)
        # 更多断言以验证结果

    def test_generate_until(self):
        requests = [
            Instance(
                request_type="generate_until",
                doc={"text": "some doc text"},
                arguments=("input text", {"max_gen_toks": 10}),
                idx=0
            )
        ]
        results = self.model.generate_until(requests)
        self.assertEqual(len(results), 1)
        # 更多断言以验证结果

if __name__ == '__main__':
    unittest.main()
