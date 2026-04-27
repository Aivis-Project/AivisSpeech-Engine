# syntax=docker/dockerfile:1.4

# TODO: build-arg と target のドキュメントをこのファイルに書く

ARG BASE_IMAGE=ubuntu:22.04
ARG BASE_RUNTIME_IMAGE=$BASE_IMAGE

# Compile Python (version locked)
FROM ${BASE_IMAGE} AS compile-python-env

ARG DEBIAN_FRONTEND=noninteractive

RUN <<EOF
    set -eux
    apt-get update
    apt-get install -y \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        curl \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        libffi-dev \
        liblzma-dev \
        git
    apt-get clean
    rm -rf /var/lib/apt/lists/*
EOF

ARG PYTHON_VERSION=3.11.9
ARG PYENV_VERSION=v2.4.11
ARG PYENV_ROOT=/tmp/.pyenv
ARG PYBUILD_ROOT=/tmp/python-build
RUN <<EOF
    set -eux

    git clone -b "${PYENV_VERSION}" https://github.com/pyenv/pyenv.git "$PYENV_ROOT"
    PREFIX="$PYBUILD_ROOT" "$PYENV_ROOT"/plugins/python-build/install.sh
    "$PYBUILD_ROOT/bin/python-build" -v "$PYTHON_VERSION" /opt/python

    rm -rf "$PYBUILD_ROOT" "$PYENV_ROOT"
EOF

# FIXME: add /opt/python to PATH
# not working: /etc/profile read only on login shell
# not working: /etc/environment is the same
# not suitable: `ENV` is ignored by docker-compose
# RUN <<EOF
#     set -eux
#     echo "export PATH=/opt/python/bin:\$PATH" > /etc/profile.d/python-path.sh
#     echo "export LD_LIBRARY_PATH=/opt/python/lib:\$LD_LIBRARY_PATH" >> /etc/profile.d/python-path.sh
#     echo "export C_INCLUDE_PATH=/opt/python/include:\$C_INCLUDE_PATH" >> /etc/profile.d/python-path.sh
#
#     rm -f /etc/ld.so.cache
#     ldconfig
# EOF


# Runtime
FROM ${BASE_RUNTIME_IMAGE} AS runtime-env
ARG DEBIAN_FRONTEND=noninteractive

# Set timezone to Asia/Tokyo
ENV TZ=Asia/Tokyo

WORKDIR /opt/aivisspeech-engine

# ca-certificates: pyopenjtalk dictionary download
# build-essential: pyopenjtalk local build
# ref: https://github.com/VOICEVOX/voicevox_engine/issues/770
RUN <<EOF
    set -eux

    apt-get update
    apt-get install -y \
        git \
        curl \
        cmake \
        ca-certificates \
        build-essential \
        gosu
    apt-get clean
    rm -rf /var/lib/apt/lists/*

    # Create a general user
    useradd --create-home user
EOF

# Copy python env
COPY --from=compile-python-env /opt/python /opt/python

# Install Python dependencies
ADD ./uv.lock ./pyproject.toml /opt/aivisspeech-engine/
RUN <<EOF
    /opt/python/bin/pip3 install uv
    chown -R user /opt/aivisspeech-engine
    # Install Rust (Sudachipy arm64 wheel build dependencies)
    gosu user bash -c "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"
    gosu user bash -c "source /home/user/.cargo/env; /opt/python/bin/uv sync --frozen --python /opt/python/bin/python3"
    gosu user rm -rf /home/user/.cargo
EOF

# Add local files
ADD ./voicevox_engine /opt/aivisspeech-engine/voicevox_engine
ADD ./docs /opt/aivisspeech-engine/docs
ADD ./run.py ./engine_manifest.json /opt/aivisspeech-engine/
ADD ./resources /opt/aivisspeech-engine/resources
ADD ./tools/generate_licenses.py /opt/aivisspeech-engine/tools/
ADD ./tools/licenses /opt/aivisspeech-engine/tools/licenses

# Replace version
ARG AIVISSPEECH_ENGINE_VERSION=latest
RUN sed -i "s/__version__ = \"latest\"/__version__ = \"${AIVISSPEECH_ENGINE_VERSION}\"/" /opt/aivisspeech-engine/voicevox_engine/__init__.py
RUN sed -i "s/\"version\": \"999\\.999\\.999\"/\"version\": \"${AIVISSPEECH_ENGINE_VERSION}\"/" /opt/aivisspeech-engine/engine_manifest.json

# pyopenjtalk-plus include dictionary in itself, download is not needed

# Create container start shell
COPY --chmod=775 <<EOF /entrypoint.sh
#!/bin/bash
set -eux

exec "\$@"
EOF

ENTRYPOINT [ "/entrypoint.sh"  ]
CMD [ "gosu", "user", ".venv/bin/python", "./run.py", "--host", "0.0.0.0" ]

# Enable use_gpu
FROM runtime-env AS runtime-nvidia-env
CMD [ "gosu", "user", ".venv/bin/python", "./run.py", "--use_gpu", "--host", "0.0.0.0" ]
