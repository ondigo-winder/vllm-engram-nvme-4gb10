# vLLM nightly (0.30.1rc1.dev558, commit 0cbac6cd1) with the Engram disk-offload patch.
# The base image must already be present locally (it is the image the cluster serves with).
ARG BASE=vllm/vllm-openai@sha256:8e6e3752946ff2dde451b313cbaec2ea8e7b52221b21bfbcf53fb68dd4f4c42e
FROM ${BASE}
ARG VLLM_DIR=/usr/local/lib/python3.12/dist-packages
COPY 0001-engram-disk-offload.patch /tmp/
RUN cd ${VLLM_DIR} && patch -p1 --forward < /tmp/0001-engram-disk-offload.patch \
 && python3 -m py_compile vllm/config/engram.py vllm/models/deepseek_v41/common/engram.py \
 && rm /tmp/0001-engram-disk-offload.patch
