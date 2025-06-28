mod vocab;
use pyo3::prelude::*;
use rayon::prelude::*;
use neuroflow::FeedForward;
use neuroflow::data::DataSet;


fn log_sigmoid(x: f64) -> f64 {
    -((1.0 + (-x).exp()).ln())
}


#[pyclass]
pub struct Model {
    pub dim: usize,
    pub negative_sample: usize,
    pub window_size: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub vocab: Option<Vocab>
    pub net: FeedForward
}

#[pymethods]
impl Model {
    pub fn new(dim: usize, negative_sample: usize, window_size: usize, epochs: usize, learning_rate: f64) -> Self {
        Model {
            dim,
            negative_sample,
            window_size,
            epochs,
            learning_rate,
            vocab: None,
            net: FeedForward::new(&[2, dim, 1])
        }
    }

    pub fn set_vocab(&mut self, vocab: Vocab) {
        self.vocab = Some(vocab);
    }

    pub fn get_vocab(&self) -> Option<HashMap<String, usize>> {
        self.vocab.word_to_id
    }

    pub fn train(&self, documents: Vec<Vec<String>>) -> PyResult<()> {
        let vocab: Vocab = Vocab::from_words(
            documents.par_iter().flat_map(|doc| doc.iter().cloned()).collect()
        );
        self.set_vocab(vocab);

        // Here you would implement the training logic using the vocab and the documents.
        // For now, we just return Ok to indicate success.
        Ok(())
    }

    pub fn predict(&self, words: Vec<String>) -> PyResult<Vec<(String, f64)>> {}
}