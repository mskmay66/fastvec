use rayon::prelude::*;
use pyo3::prelude::*;
mod vocab;

fn negative_sample(input: usize, vocab_size: usize, num_samples: usize) -> Vec<(usize, usize, u8)> {
    let mut samples = Vec::new();
    let mut rng = rand::thread_rng();
    for _ in 0..num_samples {
        let random_index = rng.gen_range(0..vocab.size);
        let random_word = vocab.valid_ids[random_index];
        if random_word != input {
            samples.push((input, random_word, 0)); // Negative sample
        }
    }
    samples
}


#[pyfunction]
pub fn build(documents: Vec<Vec<String>>, vocab: vocab::Vocab, window: Option<usize>) -> PyResult<Vec<(usize, usize, u8)>> {
    let mut examples = Vec::new();
    documents.par_iter().for_each(|doc| {
        let encoded_doc: Vec<usize> = vocab.get_ids(&doc);
        for w in encoded_doc.windows(window.unwrap_or(5)) { // Assuming context window size of 5
            let center = w[(window / 2) as u32]; // Center word in the context window
            for (i, &word) in window.iter().enumerate() {
                if i != 2 { // Skip the center word
                    examples.push((center, word, 1)); // Positive example
                    examples.extend(vocab.negative_sample(center, vocab.size 5)); // Negative samples
                }
            }
        }
    });
    Ok(examples)
}