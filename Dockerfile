FROM nvidia/cuda:13.0.1-devel-ubuntu24.04 AS build
# empty: x86-64 -> RTX 30 / 40 / 50 series (86;89;120), arm64 -> GB10 / DGX Spark (121); the engine picks kernels per GPU
ARG CUDA_ARCHITECTURES=
ARG TARGETARCH
ARG BUILD_JOBS=4
RUN apt-get update && apt-get install -y --no-install-recommends cmake ninja-build g++ && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY CMakeLists.txt ./
COPY src ./src
COPY include ./include
COPY cmake ./cmake
COPY tools ./tools
COPY tests ./tests
COPY third_party ./third_party
RUN arch="${CUDA_ARCHITECTURES}"; \
    if [ -z "$arch" ]; then case "${TARGETARCH:-$(dpkg --print-architecture)}" in arm64) arch=121 ;; *) arch="86;89;120" ;; esac; fi; \
    echo "CUDA architectures: $arch"; \
    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$arch" -DBUILD_TESTING=OFF && cmake --build build --target tqserve tqtokenize -j "${BUILD_JOBS}"

FROM nvidia/cuda:13.0.1-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/build/tqserve /usr/local/bin/tqserve
COPY --from=build /src/build/tqtokenize /usr/local/bin/tqtokenize
COPY docker/entrypoint.sh /usr/local/bin/tqllm-entrypoint
# Ship the licence and notices with the distributed image (Apache-2.0 requires the licence
# text and third-party notices travel with the binaries). These come from the build context
# directly, not the build stage, because the build stage never copied them in.
COPY LICENSE NOTICE /usr/local/share/tqllm/
COPY licenses/ /usr/local/share/tqllm/licenses/
COPY third_party/jinja/LICENSE /usr/local/share/tqllm/third_party/jinja/LICENSE
COPY third_party/jinja/NOTICE /usr/local/share/tqllm/third_party/jinja/NOTICE
RUN chmod +x /usr/local/bin/tqllm-entrypoint
EXPOSE 8086
ENTRYPOINT ["tqllm-entrypoint"]
