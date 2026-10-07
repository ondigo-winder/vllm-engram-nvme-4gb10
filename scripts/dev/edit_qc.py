import sys
root = sys.argv[1]
p = root + "/vllm/models/deepseek_v41/quant_config.py"; s = open(p).read()
old = '''            if self.expert_dtype == "fp4":
                if self.moe_quant_algo == "NVFP4":
                    from vllm.model_executor.layers.quantization.modelopt import (
                        ModelOptNvFp4FusedMoE,
                    )

                    return ModelOptNvFp4FusedMoE(
                        quant_config=self._get_nvfp4_config(),
                        moe_config=layer.moe_config,
                    )
                return Mxfp4MoEMethod(layer.moe_config)
'''
new = '''            if self.expert_dtype == "fp4":
                if self.moe_quant_algo == "NVFP4" and self._routed_experts_are_nvfp4(
                    prefix
                ):
                    from vllm.model_executor.layers.quantization.modelopt import (
                        ModelOptNvFp4FusedMoE,
                    )

                    return ModelOptNvFp4FusedMoE(
                        quant_config=self._get_nvfp4_config(),
                        moe_config=layer.moe_config,
                    )
                return Mxfp4MoEMethod(layer.moe_config)
'''
assert s.count(old) == 1; s = s.replace(old, new)
old = '''    @property
    def moe_quant_algo(self) -> str:
        self._resolve_moe_overrides()
        return self._resolved_moe_quant_algo or ""
'''
new = '''    @property
    def moe_quant_algo(self) -> str:
        self._resolve_moe_overrides()
        return self._resolved_moe_quant_algo or ""

    def _routed_experts_are_nvfp4(self, prefix: str) -> bool:
        """Whether the checkpoint stores this layer's routed experts as NVFP4.

        ModelOpt's NVFP4 export quantizes the backbone experts only: its
        ``quantized_layers`` lists ``layers.N.ffn.experts`` and ``ignore``
        holds ``mtp.*``, so the MTP/DSpark draft experts keep the original
        MXFP4 tensors (``w1.weight`` int8 + ``w1.scale`` e8m0). vLLM builds
        those draft layers as ``layers.{num_hidden_layers + i}``; give them
        the MXFP4 method or they load NVFP4 scales that are not there.
        """
        try:
            hf_config = get_current_vllm_config().model_config.hf_config
        except Exception:
            return True
        quant_cfg = getattr(hf_config, "quantization_config", None) or {}
        quantized = quant_cfg.get("quantized_layers") or {}
        if not isinstance(quantized, dict) or not quantized:
            return True
        try:
            idx = extract_layer_index(prefix)
        except Exception:
            return True
        num_layers = getattr(hf_config, "num_hidden_layers", None)
        if num_layers is not None and idx >= num_layers:
            name = f"mtp.{idx - num_layers}.ffn.experts"
        else:
            name = f"layers.{idx}.ffn.experts"
        nvfp4 = any(k == name or k.startswith(name + ".") for k in quantized)
        if not nvfp4:
            from vllm.logger import init_logger

            init_logger(__name__).info_once(
                "DeepSeek V4 routed experts %s are not NVFP4 in this "
                "checkpoint; using the MXFP4 expert method for them.",
                name,
            )
        return nvfp4
'''
assert s.count(old) == 1; s = s.replace(old, new)
s = s.replace("from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod\n", "from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod\nfrom vllm.model_executor.models.utils import extract_layer_index\n", 1)
assert "extract_layer_index" in s.split("class DeepseekV4FP8Config")[0], "import failed"
assert "get_current_vllm_config" in s.split("class DeepseekV4FP8Config")[0], "no get_current_vllm_config import"
open(p, "w").write(s); print("ok")
