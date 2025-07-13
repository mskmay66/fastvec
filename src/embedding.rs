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

    pub fn get_vector(&self, id: usize) -> PyResult<Option<Vec<f32>>> {
        match self.vectors.get(&id) {
            Some(vector) => Ok(Some(vector.clone())),
            None => Ok(None),
        }
    }

    pub fn get_vectors(&self, ids: Vec<usize>) -> PyResult<Vec<Vec<f32>>> {
        Ok(ids.par_iter().filter_map(|id| {
            self.get_vector(id.clone()).unwrap()
        }).collect())
    }
}
