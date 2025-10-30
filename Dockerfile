ARG CUDA_VERSION=12.9.1
ARG UBUNTU_VERSION=24.04

FROM docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as compile
RUN apt-get update
RUN apt-get install -y curl
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

COPY . /compile
WORKDIR /compile
ARG CUDA_COMPUTE_CAP=86
RUN . "$HOME/.cargo/env" && cargo build --release --features cuda

FROM docker.io/ubuntu:${UBUNTU_VERSION}
COPY --from=compile /compile/target/release/deepseek-ocr-cli    /usr/local/bin/deepseek-ocr-cli
COPY --from=compile /compile/target/release/deepseek-ocr-server /usr/local/bin/deepseek-ocr-server
# cudart, curand, cublas, cublasLt
COPY --from=compile /usr/local/cuda/lib64/libcudart.so.* /usr/local/cuda/lib64/
COPY --from=compile /usr/local/cuda/lib64/libcurand.so.* /usr/local/cuda/lib64/
COPY --from=compile /usr/local/cuda/lib64/libcublas.so.* /usr/local/cuda/lib64/
COPY --from=compile /usr/local/cuda/lib64/libcublasLt.so.* /usr/local/cuda/lib64/
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

ENTRYPOINT ["/usr/local/bin/deepseek-ocr-server"]
