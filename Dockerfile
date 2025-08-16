FROM rust:latest AS chef
# Use cargo-chef to prepare a recipe for caching dependencies
RUN cargo install cargo-chef

WORKDIR /app

# Create a plan (recipe) for optimized dependency caching
FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends -qq g++ pkg-config libssl-dev \
    # X-Server & Sound
    xorg xauth libx11-dev libxkbcommon-x11-0 \
    libasound2-dev libudev-dev \
    # GStreamer
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-nice gstreamer1.0-tools \
    gstreamer1.0-x gstreamer1.0-gl gstreamer1.0-vaapi gstreamer1.0-plugins-base-apps \
    gstreamer1.0-alsa gstreamer1.0-rtsp gstreamer1.0-plugins-bad-videoparsers \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstrtspserver-1.0-dev 

WORKDIR /app

# Build and cache dependencies using the recipe
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json

COPY . .

# Build just our application
RUN cargo build --release

# Runtime Stage
# https://github.com/NVIDIA/k8s-samples/blob/main/deployments/container/vulkan/Dockerfile
# https://github.com/ika-rwth-aachen/carla-simulator/blob/main/Util/Docker/vulkan-base/Vulkan.Dockerfile
# https://www.nvidia.com/en-us/drivers/
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all
ENV GST_DEBUG=3
ENV WGPU_BACKEND=vulkan
ENV XDG_RUNTIME_DIR=/tmp/xdg
ENV WGPU_POWER_PREF=high
# GSTREAMER-RELATED
ENV LADSPA_PATH=/usr/lib/ladspa
# CUDA paths
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

RUN sed -i 's/ main$/ main contrib non-free/g' /etc/apt/sources.list

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential software-properties-common ca-certificates

# runtime dependencies
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    ca-certificates pkg-config curl wget \
    # X-Server & Sound
    xorg xauth libx11-dev libxkbcommon-x11-0 \
    libasound2-dev libudev-dev \
    libxkbcommon-dev libxkbcommon0 libglib2.0-0 libxext6 \
    mesa-utils mesa-vulkan-drivers libvulkan1 vulkan-tools \
    libssl-dev \
    # LADSPA plugins (GSTREAMER-RELATED)
    ladspa-sdk \
    # VA-API support (GSTREAMER-RELATED)
    libva-drm2 \
    vainfo \
    # intel va encoder driver (GSTREAMER-RELATED)
    intel-media-va-driver \
    libmfx1 \
    # GStreamer
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-nice gstreamer1.0-tools \
    gstreamer1.0-x gstreamer1.0-gl gstreamer1.0-vaapi gstreamer1.0-plugins-base-apps \
    gstreamer1.0-alsa gstreamer1.0-rtsp gstreamer1.0-plugins-bad-videoparsers \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstrtspserver-1.0-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Vulkan SDK
# You can set VULKAN_SDK_VERSION as latest via build-arg=`curl https://vulkan.lunarg.com/sdk/latest/linux.txt`
ARG VULKAN_SDK_VERSION=1.3.268.0
ARG VULKAN_API_VERSION=1.3.268
# Download the Vulkan SDK and extract the headers, loaders, layers and binary utilities
RUN wget -q --show-progress \
    --progress=bar:force:noscroll \
    https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/linux/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.xz \
    -O /tmp/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz \
    && echo "Installing Vulkan SDK ${VULKAN_SDK_VERSION}" \
    && mkdir -p /opt/vulkan \
    && tar -xf /tmp/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz -C /opt/vulkan \
    && mkdir -p /usr/local/include/ && cp -ra /opt/vulkan/${VULKAN_SDK_VERSION}/x86_64/include/* /usr/local/include/ \
    && mkdir -p /usr/local/lib && cp -ra /opt/vulkan/${VULKAN_SDK_VERSION}/x86_64/lib/* /usr/local/lib/ \
    && cp -a /opt/vulkan/${VULKAN_SDK_VERSION}/x86_64/lib/libVkLayer_*.so /usr/local/lib \
    && mkdir -p /usr/local/share/vulkan/explicit_layer.d \
    && cp /opt/vulkan/${VULKAN_SDK_VERSION}/x86_64/etc/vulkan/explicit_layer.d/VkLayer_*.json /usr/local/share/vulkan/explicit_layer.d \
    && mkdir -p /usr/local/share/vulkan/registry \
    && cp -a /opt/vulkan/${VULKAN_SDK_VERSION}/x86_64/share/vulkan/registry/* /usr/local/share/vulkan/registry \
    && cp -a /opt/vulkan/${VULKAN_SDK_VERSION}/x86_64/bin/* /usr/local/bin \
    && ldconfig \
    && rm /tmp/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz && rm -rf /opt/vulkan

# Generate Nvidia driver config
RUN mkdir -p /etc/vulkan/icd.d && \
    echo "{" > /etc/vulkan/icd.d/nvidia_icd.json; \
    echo "    \"file_format_version\" : \"1.0.0\"," >> /etc/vulkan/icd.d/nvidia_icd.json; \
    echo "    \"ICD\": {" >> /etc/vulkan/icd.d/nvidia_icd.json; \
    echo "        \"library_path\": \"libGLX_nvidia.so.0\"," >> /etc/vulkan/icd.d/nvidia_icd.json; \
    echo "        \"api_version\" : \"${VULKAN_API_VERSION}\"" >> /etc/vulkan/icd.d/nvidia_icd.json; \
    echo "    }" >> /etc/vulkan/icd.d/nvidia_icd.json; \
    echo "}" >> /etc/vulkan/icd.d/nvidia_icd.json

COPY --from=builder /app/target/release/bevy_spatial /app/bevy_spatial

RUN mkdir -p /app/assets \
    && mkdir -p /tmp/runtime-dir \
    && chmod 777 /tmp/runtime-dir \
    && mkdir -p /tmp/xdg \
    && chmod 777 /tmp/xdg

WORKDIR /app

EXPOSE 8080 5000/udp

ENTRYPOINT [ "./bevy_spatial" ]