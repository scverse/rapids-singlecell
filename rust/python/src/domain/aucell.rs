//! aucell bindings. All CUDA allocations are borrowed from CuPy.
#![allow(non_snake_case, clippy::too_many_arguments)]
use super::*;
#[pyfunction]
#[pyo3(signature=(ranks, *, R, C, cnct, starts, lens, n_sets, n_up, max_aucs, es, stream=0))]
fn auc(
    py: Python<'_>,
    ranks: &Bound<'_, PyAny>,
    R: u64,
    C: u64,
    cnct: &Bound<'_, PyAny>,
    starts: &Bound<'_, PyAny>,
    lens: &Bound<'_, PyAny>,
    n_sets: u64,
    n_up: i32,
    max_aucs: &Bound<'_, PyAny>,
    es: &Bound<'_, PyAny>,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;

    let ranks = read(Some(ranks), &cupy, "ranks", "int", false)?;
    let cnct = read(Some(cnct), &cupy, "cnct", "int", false)?;
    let starts = read(Some(starts), &cupy, "starts", "int", false)?;
    let lens = read(Some(lens), &cupy, "lens", "int", false)?;
    let max_aucs = read(Some(max_aucs), &cupy, "max_aucs", "float", false)?;
    let es = read(Some(es), &cupy, "es", "float", false)?;
    let input_elements = R
        .checked_mul(C)
        .ok_or_else(|| PyValueError::new_err("AUCell input dimensions overflow"))?;
    let output_elements = R
        .checked_mul(n_sets)
        .ok_or_else(|| PyValueError::new_err("AUCell output dimensions overflow"))?;
    if input_elements > ranks.len()
        || output_elements > es.len()
        || n_sets > starts.len()
        || n_sets > lens.len()
        || n_sets > max_aucs.len()
    {
        return Err(PyValueError::new_err(
            "AUCell dimensions exceed an allocation",
        ));
    }
    for input in [&ranks, &cnct, &starts, &lens, &max_aucs] {
        es.array
            .as_ref()
            .unwrap()
            .require_disjoint(input.array.as_ref().unwrap())?;
    }
    launch(
        &cupy,
        &[&ranks, &cnct, &starts, &lens, &max_aucs, &es],
        "domain_aucell_auc",
        output_elements,
        stream,
        vec![
            ranks.pointer(),
            R,
            C,
            cnct.pointer(),
            starts.pointer(),
            lens.pointer(),
            n_sets,
            n_up as u64,
            max_aucs.pointer(),
            es.pointer(),
        ],
    )?;
    Ok(())
}
pub(super) fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = submodule(parent, "_aucell_cuda")?;
    m.add_function(wrap_pyfunction!(auc, &m)?)?;
    Ok(())
}
