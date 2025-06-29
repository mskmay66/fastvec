use pyo3::prelude::*;


mod vocab;
mod dataset;
use vocab::Vocab;
use dataset::build;

#[pymodule]
fn fastvec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Vocab>()?;
    m.add_function(wrap_pyfunction!(build, m)?)?;
    Ok(())
}
