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


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_creation() {
        let mut embedding = Embedding::new(3);
        assert_eq!(embedding.dim, 3);
        assert!(embedding.vectors.is_empty());
    }

    #[test]
    fn test_add_vector() {
        let mut embedding = Embedding::new(3);
        embedding.add_vector(1, vec![0.1, 0.2, 0.3]).unwrap();
        assert_eq!(embedding.vectors.len(), 1);
        assert_eq!(embedding.vectors.get(&1), Some(&vec![0.1, 0.2, 0.3]));

        let result = embedding.add_vector(2, vec![0.1, 0.2]); // Should fail due to dimension mismatch
        assert!(result.is_err());

        let result = embedding.add_vector(2, vec![0.1, 0.2, 0.3]);
        assert!(result.is_ok());
        assert_eq!(embedding.vectors.len(), 2);
        assert_eq!(embedding.vectors.get(&2), Some(&vec![0.1, 0.2, 0.3]));
    }

    #[test]
    fn test_add_vectors() {
        let mut embedding = Embedding::new(3);
        let ids = vec![1, 2, 3];
        let vectors = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6], vec![0.7, 0.8, 0.9]];

        embedding.add_vectors(ids.clone(), vectors).unwrap();
        assert_eq!(embedding.vectors.len(), 3);

        for (id, vector) in ids.iter().zip(vectors) {
            assert_eq!(embedding.vectors.get(id), Some(&vector));
        }

        let result = embedding.add_vectors(vec![4], vec![vec![0.1, 0.2]]); // Should fail due to dimension mismatch
        assert!(result.is_err());
    }

    #[test]
    fn test_get_vector() {
        let mut embedding = Embedding::new(3);
        embedding.add_vector(1, vec![0.1, 0.2, 0.3]).unwrap();

        let vector = embedding.get_vector(1).unwrap();
        assert_eq!(vector, Some(vec![0.1, 0.2, 0.3]));

        let vector = embedding.get_vector(2).unwrap();
        assert_eq!(vector, None);
    }

    #[test]
    fn test_get_vectors() {
        let mut embedding = Embedding::new(3);
        embedding.add_vector(1, vec![0.1, 0.2, 0.3]).unwrap();
        embedding.add_vector(2, vec![0.4, 0.5, 0.6]).unwrap();

        let vectors = embedding.get_vectors(vec![1, 2, 3]).unwrap();
        assert_eq!(vectors.len(), 2);
        assert_eq!(vectors[0], vec![0.1, 0.2, 0.3]);
        assert_eq!(vectors[1], vec![0.4, 0.5, 0.6]);
        let vectors = embedding.get_vectors(vec![3, 4]).unwrap();
        assert!(vectors.is_empty()); // IDs 3 and 4 do not exist
        assert_eq!(embedding.get_vectors(vec![]).unwrap(), vec![]); // Empty input should return empty vector
    }
}
