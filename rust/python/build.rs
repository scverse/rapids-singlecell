use std::{collections::BTreeSet, env, fmt::Write, fs, path::PathBuf};

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
    let directory = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    // CUDA reads the kernel's parameter count from the image, not the host
    // argument slice. Cache its arity so stale internal call sites fail with a
    // Python error instead of letting the driver read beyond that host slice.
    let mut abi = String::from("fn kernel_arity(name: &str) -> Option<usize> { match name {\n");
    for entry in ptx.split(".entry ").skip(1) {
        let (name, tail) = entry.split_once('(').expect("PTX entry signature");
        let parameters = tail.split_once(')').expect("PTX parameter list").0;
        writeln!(
            abi,
            "{:?} => Some({}),",
            name.trim(),
            parameters.matches(".param ").count()
        )
        .unwrap();
    }
    abi.push_str("_ => None, } }\n");
    fs::write(directory.join("kernel_abi.rs"), abi).expect("failed to emit PTX ABI");
    fs::write(directory.join("kernels.ptx"), ptx).expect("failed to embed cuda-oxide PTX");
}
