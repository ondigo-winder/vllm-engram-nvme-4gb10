import sys
p = sys.argv[1]; s = open(p).read()
old = '''    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Map HF names into this wrapper's namespace up front and sort, so
        # the "language_model." group reaches the child loader as one
        # contiguous block (AutoWeightsLoader delegates per contiguous group,
        # and the child's load_weights finalizes fused expert weights, which
        # must not run on a partially loaded model).
        mapped = sorted(self.hf_to_vllm_mapper.apply(weights), key=lambda x: x[0])
        loader = AutoWeightsLoader(self)
        loaded_params = loader.load_weights(mapped)
'''
new = '''    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The "language_model." group must reach the child loader as one
        # contiguous block (AutoWeightsLoader delegates per contiguous group,
        # and the child's load_weights finalizes fused expert weights, which
        # must not run on a partially loaded model). Stream that group instead
        # of materializing the whole checkpoint: a sorted list keeps every
        # safetensors mapping alive until the end of the load, and the pages a
        # device copy pins out of a private file mapping are only released when
        # the mapping closes, which does not fit on a 128 GB unified-memory
        # node. Only the (small) remaining tensors are buffered.
        mapped = self.hf_to_vllm_mapper.apply(weights)
        others: list[tuple[str, torch.Tensor]] = []

        def language_model_stream() -> Iterable[tuple[str, torch.Tensor]]:
            for name, tensor in mapped:
                if name.startswith("language_model."):
                    yield name, tensor
                else:
                    others.append((name, tensor))

        loader = AutoWeightsLoader(self)
        loaded_params = loader.load_weights(language_model_stream())
        if others:
            loaded_params |= loader.load_weights(sorted(others, key=lambda x: x[0]))
'''
assert old in s, "context not found"
s = s.replace(old, new); open(p, "w").write(s); print("ok")
