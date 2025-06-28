use rayon::prelude::*;


pub struct Vocab {
    pub size: usize,
    pub words: Vec<String>,
    pub word_to_id: std::collections::HashMap<String, usize>,
    pub valid_ids: Vec<usize>,
}

impl Vocab {
    pub fn new() -> Self {
        Vocab {
            size: 0,
            words: Vec::new(),
            word_to_id: std::collections::HashMap::new(),
            valid_ids: Vec::new(),
        }
    }

    pub fn from_words(words: Vec<String>) -> Self {
        let mut vocab = Vocab::new();
        words.par_iter().for_each(|word| {
            vocab.add_word(word);
        });
        self.valid_ids = vocab.subsample();
        vocab
    }

    pub fn get_ids(&self, words: Vec<String>) -> Vec<usize> {
        words.par_iter().map(|word| self.word_to_id[word]).collect()
    }

    pub fn add_word(&mut self, word: String) {
        if !self.word_to_id.contains_key(&word) {
            let id = self.size;
            self.word_to_id.insert(word.clone(), id);
            self.words.push(word);
            self.size += 1;
        }
    }

    pub fn get_id(&self, word: &str) -> Option<usize> {
        self.word_to_id.get(word).cloned()
    }

    fn subsample(&self) -> Vec<usize> {
        let n: usize = self.size;
        let mut word_to_freq: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
        lst.par_iter().for_each(|word| { *word_to_freq.entry(*word).or_insert(0 as f64) += 1 as f64 });
        word_to_freq.par_iter().for_each(|(_, val)| *val /= n as f64);
        let mut rng = rand::thread_rng();
        let indices: Vec<usize> = word_to_freq.par_iter().map(|k,v| { 
            let p: f64 = ((v / 0.001).sqrt() + 1) * (0.001 / v);
            let r: f64 = rng.gen();
            if r < p { *k }
        }).collect();
        indices
    }
}