from . import (
    anthropic_llms,
    dummy,
    gguf,
    huggingface,
    mamba_lm,
    nemo_lm,
    neuralmagic,
    neuron_optimum,
    openai_completions,
    optimum_lm,
    textsynth,
    vllm_causallms,
)

# 导入 LM 基类
from lm_eval.api.model import LM

# 导入您的模型
from .vicuna_with_token_perturbation import VicunaWithTokenPerturbation

# 注册您的模型
from lm_eval.api.registry import register_model

@register_model("vicuna_with_token_perturbation")
class VicunaWithTokenPerturbation(LM):
    pass  # 如果已经在 vicuna_with_token_perturbation.py 中实现了类，这里可以不用实现具体方法

# TODO: implement __all__

try:
    # enable hf hub transfer if available
    import hf_transfer  # type: ignore # noqa
    import huggingface_hub.constants  # type: ignore

    huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True
except ImportError:
    pass
