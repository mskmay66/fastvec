use rayon::prelude::*;
use pyo3::prelude::*;


#[pyclass]
pub struct Vocab {
    pub size: usize,
    pub words: Vec<String>,
    pub word_to_id: std::collections::HashMap<String, usize>,
}

#[pymethods]
impl Vocab {
    #[new]
    pub fn new() -> Self {
        Vocab {
            size: 0,
            words: Vec::new(),
            word_to_id: std::collections::HashMap::new(),
        }
    }

    #[staticmethod]
    pub fn from_words(words: Vec<String>) -> Self {
        let mut vocab = Vocab::new();
        let mut word_to_freq: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
        words.par_iter().for_each(|word| {
            vocab.add_word(word);
            word_to_freq
                .entry(vocab.word_to_id[word])
                .and_modify(|e| *e += 1.0)
                .or_insert(1.0);
        });
        vocab.word_to_id = words
            .par_iter()
            .map(|word, idx| (word, vocab.subsample(word, &word_to_freq)))
            .collect();
        vocab
    }

    pub fn get_ids(&self, words: Vec<String>) -> PyResult<Vec<usize>> {
        Ok(words.par_iter().map(|word| self.word_to_id[word]).collect())
    }

    pub fn add_word(&mut self, word: String) -> PyResult<()> {
        if !self.word_to_id.contains_key(&word) {
            let id = self.size;
            self.word_to_id.insert(word.clone(), id);
            self.words.push(word);
            self.size += 1;
        }
        Ok(())
    }

    pub fn get_id(&self, word: &str) -> PyResult<Option<usize>> {
        Ok(self.word_to_id.get(word).cloned())
    }

    fn subsample(&self, word: &str, word_to_freq: &HashMap<usize, f64>) -> Option>usize> {
        if let Some(&id) = self.word_to_id.get(word) {
            let freq = word_to_freq.get(&id).unwrap_or(&0.0);
            let p = ((freq / 0.001).sqrt() + 1.0) * (0.001 / freq);
            let r: f64 = rand::random();
            if r < p {
                return Some(id);
            }
        }
        None
    }
}