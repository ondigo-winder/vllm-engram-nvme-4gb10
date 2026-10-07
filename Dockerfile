# vLLM nightly (0.30.1rc1.dev558, commit 0cbac6cd1) with the Engram disk-offload patch.
# The base image must already be present locally (it is the image the cluster serves with).
ARG BASE=vllm/vllm-openai@sha256:8e6e3752946ff2dde451b313cbaec2ea8e7b52221b21bfbcf53fb68dd4f4c42e
FROM ${BASE}
ARG VLLM_DIR=/usr/local/lib/python3.12/dist-packages
# SM120=1 also applies the experimental sm_120 (DGX Spark / GB10) page-size
# workarounds for DeepSeek-V4.1, see README "Status on GB10".
ARG SM120=1
COPY 0001-engram-disk-offload.patch 0002-sm120-deepseek-v41-page-sizes.patch /tmp/
RUN cd ${VLLM_DIR} && patch -p1 --forward < /tmp/0001-engram-disk-offload.patch \
 && if [ "${SM120}" = "1" ]; then patch -p1 --forward < /tmp/0002-sm120-deepseek-v41-page-sizes.patch; fi \
 && python3 -m py_compile vllm/config/engram.py vllm/models/deepseek_v41/common/engram.py \
      vllm/model_executor/model_loader/weight_utils.py vllm/models/deepseek_v41/nvidia/vl_model.py \
      vllm/models/deepseek_v41/attention.py vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py \
      vllm/v1/attention/backends/mla/indexer.py vllm/models/deepseek_v41/quant_config.py \
 && rm /tmp/0001-engram-disk-offload.patch /tmp/0002-sm120-deepseek-v41-page-sizes.patch
