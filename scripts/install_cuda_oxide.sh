#!/usr/bin/env bash
# Install exactly the compiler revision pinned by the device crate.
# Run from a checkout or source distribution with CUDA 13, Clang and libclang.
set -euo pipefail
rsc_source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
rsc_toolchain=$(sed -n 's/^channel = "\([^"]*\)"/\1/p' "${rsc_source_dir}/rust/rust-toolchain.toml")
rsc_oxide_rev=$(sed -n 's/.*rev = "\([^"]*\)".*/\1/p' "${rsc_source_dir}/rust/kernels/Cargo.toml")
if [[ -z "${rsc_toolchain}" || ! "${rsc_oxide_rev}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Unable to read the pinned Rust/cuda-oxide toolchain" >&2
    exit 1
fi
if ! command -v rustup >/dev/null 2>&1; then
    rsc_installer=$(mktemp)
    trap 'rm -f -- "${rsc_installer}"' EXIT
    curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs -o "${rsc_installer}"
    sh "${rsc_installer}" -y --profile minimal --default-toolchain none
    export PATH="${CARGO_HOME:-${HOME}/.cargo}/bin:${PATH}"
fi
rustup toolchain install "${rsc_toolchain}" --profile minimal \
    --component rust-src --component rustc-dev --component llvm-tools \
    --component rustfmt --component clippy
cargo +"${rsc_toolchain}" install --git https://github.com/NVlabs/cuda-oxide.git \
    --rev "${rsc_oxide_rev}" --locked --force cargo-oxide
