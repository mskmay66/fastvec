use pyo3::prelude::*;
mod vocab;
mod dataset;

#[pymodule]
fn fastvec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<vocab::Vocab>()?;
    m.add_function(wrap_pyfunction!(dataset::build, m)?)?;
    Ok(())
}
