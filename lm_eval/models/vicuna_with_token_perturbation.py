import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance

class TokenPerturbation:
    def __init__(self, perturbation_level=0.1):
        self.perturbation_level = perturbation_level
    
    def perturb(self, prompt):
        tokens = prompt.split()
        num_tokens_to_perturb = int(len(tokens) * self.perturbation_level)
        for _ in range(num_tokens_to_perturb):
            idx = random.randint(0, len(tokens) - 1)
            tokens[idx] = self._perturb_token(tokens[idx])
        return ' '.join(tokens)
    
    def _perturb_token(self, token):
        return token[::-1]  # reversing the token

class VicunaWithTokenPerturbation(LM):
    def __init__(self, model_name='lmsys/vicuna-7b-v1.5', perturbation_level=0.1):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.defense_method = TokenPerturbation(perturbation_level)

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        results = []
        for req in requests:
            input_str, target_str = req.arguments
            input_str = self.defense_method.perturb(input_str)
            ll = self._calculate_loglikelihood(input_str, target_str)
            is_greedy = self._is_greedy(input_str, target_str)
            results.append((ll, is_greedy))
        return results

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float]]:
        results = []
        for req in requests:
            input_str = req.arguments[0]
            input_str = self.defense_method.perturb(input_str)
            ll = self._calculate_loglikelihood_rolling(input_str)
            results.append((ll,))
        return results

    def generate_until(self, requests: list[Instance]) -> list[str]:
        results = []
        for req in requests:
            input_str, generation_params = req.arguments
            input_str = self.defense_method.perturb(input_str)
            output = self._generate_text(input_str, generation_params)
            results.append(output)
        return results

    def _calculate_loglikelihood(self, input_str: str, target_str: str) -> float:
        inputs = self.tokenizer(input_str, return_tensors="pt")
        target_ids = self.tokenizer(target_str, return_tensors="pt").input_ids
        with torch.no_grad():
            outputs = self.model(**inputs, labels=target_ids)
        loss = outputs.loss
        if loss is not None:
            return -loss.item() * target_ids.size(1)  # returning negative log-likelihood
        else:
            return float('inf')  # or another appropriate default value

    def _is_greedy(self, input_str: str, target_str: str) -> bool:
        return True  # 示例值

    def _calculate_loglikelihood_rolling(self, input_str: str) -> float:
        inputs = self.tokenizer(input_str, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs.input_ids)
        loss = outputs.loss
        if loss is not None:
            return -loss.item() * inputs.input_ids.size(1)  # returning negative log-likelihood
        else:
            return float('inf')  # or another appropriate default value

    def _generate_text(self, input_str: str, generation_params: dict) -> str:
        inputs = self.tokenizer(input_str, return_tensors="pt")
        max_gen_toks = generation_params.pop("max_gen_toks", None)
        if max_gen_toks:
            generation_params["max_new_tokens"] = max_gen_toks
        outputs = self.model.generate(**inputs, **generation_params)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
