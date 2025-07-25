use pyo3::prelude::*;

mod vocab;
mod embedding;
mod build;
mod preprocessing;
mod word2vec;
mod doc2vec;

use vocab::Vocab;
use embedding::Embedding;
use build::Builder;
use build::Example;
use preprocessing::simple_preprocessing;
use word2vec::_Word2Vec;
use doc2vec::DocumentLayer;


#[pymodule]
fn fastvec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Vocab>()?;
    m.add_class::<Embedding>()?;
    m.add_class::<Builder>()?;
    m.add_class::<Example>()?;
    m.add_class::<preprocessing::Tokens>()?;
    m.add_function(wrap_pyfunction!(simple_preprocessing, m)?)?;
    m.add_class::<_Word2Vec>()?;
    m.add_class::<DocumentLayer>()?;
    Ok(())
}
