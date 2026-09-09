use std::{collections::BTreeSet, env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=RSC_CUDA_OXIDE_PTX");
    let source = PathBuf::from(env::var_os("RSC_CUDA_OXIDE_PTX").expect(
        "RSC_CUDA_OXIDE_PTX must name the PTX built by cuda-oxide; use the repository CMake build",
    ));
    println!("cargo:rerun-if-changed={}", source.display());
    let ptx = fs::read_to_string(&source).expect("failed to read cuda-oxide PTX");
    println!("cargo:rerun-if-changed=../kernels/exports.txt");
    let expected: BTreeSet<_> = include_str!("../kernels/exports.txt").lines().collect();
    let actual: BTreeSet<_> = ptx
        .split(".entry ")
        .skip(1)
        .map(|entry| entry.split('(').next().unwrap().trim())
        .collect();
    assert!(
        actual == expected,
        "PTX kernel exports do not match kernels/exports.txt; missing: {:?}; unexpected: {:?}",
        expected.difference(&actual).collect::<Vec<_>>(),
        actual.difference(&expected).collect::<Vec<_>>()
    );
    let destination = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("kernels.ptx");
    fs::write(destination, ptx).expect("failed to embed cuda-oxide PTX");
}
