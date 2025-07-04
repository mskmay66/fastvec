use rayon::prelude::*;
use pyo3::prelude::*;
use rand::prelude::*;
mod vocab;
mod embedding;
mod build;
use vocab::Vocab;
use embedding::Embedding;
use build::Builder;


#[pymodule]
fn fastvec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Vocab>()?;
    m.add_class::<Embedding>()?;
    m.add_class::<Builder>()?;
    // m.add_function(wrap_pyfunction!(build_w2v_training, m)?)?;
    Ok(())
}
