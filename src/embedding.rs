use rayon::prelude::*;
use pyo3::prelude::*;
use std::collections::HashMap;


#[pyclass]
pub struct Embedding {
    pub dim: usize,
    pub vectors: HashMap<usize, Vec<f32>>,
}


#[pymethods]
impl Embedding {
    #[new]
    pub fn new(dim: usize) -> Self {
        Embedding {
            dim,
            vectors: HashMap::new(),
        }
    }

    pub fn add_vector(&mut self, id: usize, vector: Vec<f32>) -> PyResult<()> {
        if vector.len() != self.dim {
            return Err(pyo3::exceptions::PyValueError::new_err("Vector dimension mismatch"));
        }
        self.vectors.insert(id, vector);
        Ok(())
    }

    pub fn add_vectors(&mut self, ids: Vec<usize>, vectors: Vec<Vec<f32>>) -> PyResult<()> {
        if ids.len() != vectors.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("IDs and vectors length mismatch"));
        }
        ids.iter().zip(vectors).try_for_each(|(id, vector)| {
            self.add_vector(*id, vector)
        })?;
        Ok(())
    }

    pub fn get_vector(&self, id: usize) -> Option<&Vec<f32>> {
        self.vectors.get(&id)
    }

    pub fn get_vectors(&self, ids: Vec<usize>) -> PyResult<Vec<Vec<f32>>> {
        ids.par_iter().map(|id| {
            match self.get_vector(id.clone()) {
                Some(vector) => Ok(vector.clone()),
                None => Err(pyo3::exceptions::PyKeyError::new_err("Vector not found for the given ID or word")),
            }
        }).collect()
    }
}