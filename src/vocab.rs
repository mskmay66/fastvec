use pyo3::prelude::*;
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

#[pyclass]
#[derive(Clone, Debug)]
pub struct Vocab {
    #[pyo3(get)]
    pub size: usize,
    #[pyo3(get)]
    pub words: Vec<String>,
    #[pyo3(get)]
    pub word_to_id: HashMap<String, usize>,
    #[pyo3(get)]
    pub valid_ids: Vec<usize>,
    #[pyo3(get)]
    pub min_count: usize,
}

#[pymethods]
impl Vocab {
    #[new]
    pub fn new() -> Self {
        Vocab {
            size: 0,
            words: Vec::new(),
            word_to_id: HashMap::new(),
            valid_ids: Vec::new(),
            min_count: 5, // default minimum count for subsampling
        }
    }

    #[staticmethod]
    pub fn from_words(words: Vec<String>, min_count: usize) -> Self {
        let mut vocab = Vocab::new();
        vocab.min_count = min_count;
        let mut word_to_freq: HashMap<usize, f64> = HashMap::new();
        words.iter().for_each(|word| {
            let _ = vocab.add_word(word);
            word_to_freq
                .entry(vocab.word_to_id[word])
                .and_modify(|e| *e += 1.0)
                .or_insert(1.0);
        });
        let n = vocab.size;
        vocab.valid_ids = words
            .par_iter()
            .filter_map(|word| {
                if let Some(id) = vocab.word_to_id.get(word) {
                    if words.len() > 100 {
                        // only subsample larger vocabularies
                        subsample(id, &word_to_freq, n, vocab.min_count)
                    } else {
                        Some(*id)
                    }
                } else {
                    None
                }
            })
            .collect();
        vocab
    }

    pub fn get_ids(&self, words: Vec<String>) -> PyResult<Vec<usize>> {
        Ok(words
            .par_iter()
            .filter_map(|word| self.get_id(word).unwrap())
            .collect())
    }

    pub fn add_word(&mut self, word: &str) -> PyResult<()> {
        if !self.word_to_id.contains_key(word) {
            let id = self.size;
            self.word_to_id.insert(word.to_string(), id);
            self.words.push(word.to_string());
            self.size += 1;
            self.valid_ids.push(id);
        }
        Ok(())
    }

    pub fn get_id(&self, word: &str) -> PyResult<Option<usize>> {
        match self.word_to_id.get(word) {
            Some(&id) => Ok(Some(id)),
            None => Ok(None),
        }
    }

    pub fn get_random_id(&self, avoid: Option<Vec<usize>>) -> PyResult<usize> {
        let mut rng = rand::rng();
        let avoid_ids = avoid.unwrap_or(Vec::new());
        let mut max_iter = 0;
        let id = loop {
            let random_index = self
                .valid_ids
                .choose(&mut rng)
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No valid IDs available"))?
                .clone();
            if !&avoid_ids.contains(&random_index) {
                break random_index;
            }
            max_iter += 1;
            if max_iter > self.size {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "No valid random ID found after maximum iterations",
                ));
            }
        };
        Ok(id)
    }

    pub fn __len__(&self) -> PyResult<usize> {
        Ok(self.size)
    }
}

fn subsample(
    token_id: &usize,
    word_to_freq: &HashMap<usize, f64>,
    vocab_size: usize,
    min_count: usize,
) -> Option<usize> {
    let min_count = min_count as f64;
    let count = word_to_freq.get(token_id).unwrap_or(&min_count);
    if *count <= min_count {
        return None; // Skip subsampling for low-frequency words
    }

    let freq = count / vocab_size as f64;
    let p = 1.0 - (0.0001 / freq).sqrt();
    let r: f64 = rand::random();
    if r > p {
        return Some(*token_id);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab_creation() {
        let empty_vocab = Vocab::new();
        assert_eq!(empty_vocab.size, 0);
        assert!(empty_vocab.words.is_empty());
        assert!(empty_vocab.word_to_id.is_empty());
        assert!(empty_vocab.valid_ids.is_empty());

        let words = vec![
            "apple".to_string(),
            "banana".to_string(),
            "apple".to_string(),
        ];
        let vocab = Vocab::from_words(words, 5);
        assert_eq!(vocab.size, 2);
        assert_eq!(vocab.words, vec!["apple", "banana"]);
        assert_eq!(vocab.word_to_id.get("apple"), Some(&0));
        assert_eq!(vocab.word_to_id.get("banana"), Some(&1));
    }

    #[test]
    fn test_add_word() {
        let mut vocab = Vocab::new();
        assert!(vocab.add_word("test").is_ok());
        assert_eq!(vocab.size, 1);
        assert_eq!(vocab.words, vec!["test"]);
        assert_eq!(vocab.word_to_id.get("test"), Some(&0));

        // Adding the same word again should not change the size
        assert!(vocab.add_word("test").is_ok());
        assert_eq!(vocab.size, 1);
    }

    #[test]
    fn test_get_id() {
        let mut vocab = Vocab::new();
        vocab.add_word("apple").unwrap();
        vocab.add_word("banana").unwrap();

        assert_eq!(vocab.get_id("apple").unwrap(), Some(0));
        assert_eq!(vocab.get_id("banana").unwrap(), Some(1));
        assert_eq!(vocab.get_id("cherry").unwrap(), None); // Cherry not added
    }

    #[test]
    fn test_get_ids() {
        let words = vec![
            "apple".to_string(),
            "banana".to_string(),
            "cherry".to_string(),
        ];
        let mut vocab = Vocab::new();
        for word in &words {
            vocab.add_word(word).unwrap();
        }
        let ids = vocab.get_ids(words).unwrap();
        assert_eq!(ids, vec![0, 1, 2]); // Assuming apple=0, banana=1, cherry=2
    }

    #[test]
    fn test_get_random_id() {
        let mut vocab = Vocab::new();
        vocab.add_word("apple").unwrap();
        vocab.add_word("banana").unwrap();
        vocab.add_word("cherry").unwrap();

        let id = vocab.get_random_id(None).unwrap();
        assert!(id < vocab.size); // Should return a valid ID

        let id2 = vocab.get_random_id(Some(vec![id])).unwrap();
        assert!(id2 < vocab.size && id2 != id); // Should return a different valid ID
    }

    #[test]
    fn test_subsample() {
        let mut word_to_freq = HashMap::new();
        word_to_freq.insert(0, 6.0);
        word_to_freq.insert(1, 6.0);
        word_to_freq.insert(2, 10.0);

        let vocab_size = 3000000;
        assert!(subsample(&0, &word_to_freq, vocab_size, 5).is_some());
        assert!(subsample(&1, &word_to_freq, vocab_size, 5).is_some());
        assert!(subsample(&2, &word_to_freq, vocab_size, 5).is_some());

        // Test with a frequency that should not be sampled
        word_to_freq.insert(3, 1.0);
        assert!(subsample(&3, &word_to_freq, vocab_size + 1, 5).is_none());
    }

    #[test]
    fn no_dups() {
        let words = vec![
            "apple".to_string(),
            "banana".to_string(),
            "apple".to_string(),
            "cherry".to_string(),
            "banana".to_string(),
            "cherry".to_string(),
        ];
        let vocab = Vocab::from_words(words, 5);
        assert_eq!(vocab.size, 3);
        assert_eq!(vocab.words, vec!["apple", "banana", "cherry"]);
        assert_eq!(vocab.word_to_id.get("apple"), Some(&0));
        assert_eq!(vocab.word_to_id.get("banana"), Some(&1));
        assert_eq!(vocab.word_to_id.get("cherry"), Some(&2));
    }
}
