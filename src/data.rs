use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct Dataset {
    #[pyo3(get)]
    pub input_words: Vec<f32>,
    #[pyo3(get)]
    pub context_words: Vec<f32>,
    #[pyo3(get)]
    pub labels: Vec<u32>,
}

#[pymethods]
impl Dataset {
    #[new]
    pub fn new(input_words: Vec<f32>, context_words: Vec<f32>, labels: Vec<u32>) -> Self {
        Dataset {
            input_words,
            context_words,
            labels,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.input_words.is_empty() && self.context_words.is_empty() && self.labels.is_empty()
    }

    pub fn __len__(&self) -> usize {
        self.input_words.len()
    }

    pub fn __getitem__(&self, idx: usize) -> PyResult<(f32, f32, u32)> {
        if idx >= self.input_words.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "Index out of range",
            ));
        }
        Ok((
            self.input_words[idx],
            self.context_words[idx],
            self.labels[idx],
        ))
    }

    pub fn extend(&mut self, other: Dataset) -> PyResult<()> {
        self.input_words.extend(other.input_words);
        self.context_words.extend(other.context_words);
        self.labels.extend(other.labels);
        Ok(())
    }
}

pub struct DataLoader {
    pub input_words: Array2<f32>,
    pub context_words: Array2<f32>,
    pub labels: Array1<f32>,
    batch_size: usize,
}

impl DataLoader {
    pub fn new(
        input_words: Array2<f32>,
        context_words: Array2<f32>,
        labels: Array1<u32>,
        batch_size: usize,
    ) -> Self {
        DataLoader {
            input_words: input_words,
            context_words: context_words,
            labels: labels.mapv(|x| x as f32), // Convert labels to f32 for consistency
            batch_size: batch_size,
        }
    }

    pub fn from_dataset(dataset: &Dataset, batch_size: usize) -> Self {
        let input_words =
            Array2::from_shape_vec((dataset.input_words.len(), 1), dataset.input_words.clone())
                .unwrap();
        let context_words = Array2::from_shape_vec(
            (dataset.context_words.len(), 1),
            dataset.context_words.clone(),
        )
        .unwrap();
        let labels = Array1::from_vec(dataset.labels.clone()).mapv(|x| x as f32);
        DataLoader {
            input_words,
            context_words,
            labels,
            batch_size,
        }
    }

    pub fn iter(&self) -> DataLoaderIter {
        DataLoaderIter::new(self)
    }
}

pub struct DataLoaderIter<'a> {
    dataloader: &'a DataLoader,
    curr: usize,
    next: usize,
}

impl<'a> DataLoaderIter<'a> {
    pub fn new(dataloader: &'a DataLoader) -> Self {
        DataLoaderIter {
            dataloader,
            curr: 0,
            next: 0,
        }
    }
}

impl<'a> Iterator for DataLoaderIter<'a> {
    type Item = (
        ArrayView2<'a, f32>,
        ArrayView2<'a, f32>,
        ArrayView1<'a, f32>,
    );

    fn next(&mut self) -> Option<Self::Item> {
        if self.curr >= self.dataloader.input_words.shape()[0] {
            return None; // No more batches available
        }
        if self.curr + self.dataloader.batch_size > self.dataloader.input_words.shape()[0] {
            self.next = self.dataloader.input_words.shape()[0];
        } else {
            self.next = self.curr + self.dataloader.batch_size;
        }
        let input_batch = self
            .dataloader
            .input_words
            .slice(s![self.curr..self.next, ..]);
        let context_batch = self
            .dataloader
            .context_words
            .slice(s![self.curr..self.next, ..]);
        let labels_batch = self.dataloader.labels.slice(s![self.curr..self.next]);
        self.curr = self.next;
        Some((input_batch, context_batch, labels_batch))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_dataset_creation() {
        let dataset = Dataset::new(vec![1.0, 2.0], vec![3.0, 4.0], vec![1, 0]);
        assert_eq!(dataset.input_words.len(), 2);
        assert_eq!(dataset.context_words.len(), 2);
        assert_eq!(dataset.labels.len(), 2);
    }

    #[test]
    fn test_dataloader_from_dataset() {
        let dataset = Dataset::new(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], vec![1, 0, 1]);
        let dataloader = DataLoader::from_dataset(&dataset, 2);
        assert_eq!(dataloader.input_words.shape(), &[3, 1]);
        assert_eq!(dataloader.context_words.shape(), &[3, 1]);
        assert_eq!(dataloader.labels.shape(), &[3,]);
        assert_eq!(dataloader.batch_size, 2);
    }

    #[test]
    fn test_dataloader_iterator() {
        let dataset = Dataset::new(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], vec![1, 0, 1]);
        let mut dataloader = DataLoader::from_dataset(&dataset, 2);
        let mut iter = dataloader.iter();
        if let Some((input_batch, context_batch, labels_batch)) = iter.next() {
            assert_eq!(input_batch.shape(), &[2, 1]);
            assert_eq!(context_batch.shape(), &[2, 1]);
            assert_eq!(labels_batch.shape(), &[2,]);
        } else {
            panic!("Expected a batch but got None");
        }
    }
}
