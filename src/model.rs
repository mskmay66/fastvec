mod vocab;
use pyo3::prelude::*;
use rayon::prelude::*;
use neuroflow::FeedForward;
use neuroflow::data::DataSet;


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
            net: FeedForward::new(&[window_size, dim, window_size])
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

        let mut data: DataSet = DataSet::new();
        (0..documents.len()).into_par_iter().for_each(|i| {
            let doc = &documents[i];
            let ids: Vec<usize> = self.vocab.as_ref().unwrap().get_ids(doc.clone());
            data.push(&ids, &ids);
        });

        
        // Here you would implement the training logic using the vocab and the documents.
        // For now, we just return Ok to indicate success.
        Ok(())
    }

    pub fn predict(&self, words: Vec<String>) -> PyResult<Vec<(String, f64)>> {}
}