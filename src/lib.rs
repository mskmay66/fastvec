use rayon::prelude::*;
use pyo3::prelude::*;
use rand::prelude::*;
mod vocab;
use vocab::Vocab;

fn negative_sample(input: usize, vocabulary: Vocab, num_samples: usize) -> Vec<(usize, usize, u8)> {
    let mut samples = Vec::new();
    let mut rng = rand::thread_rng();
    for _ in 0..num_samples {
        let random_index = rng.gen_range(0..(vocabulary.size));
        let random_word = vocabulary.valid_ids[random_index];
        if random_word != input {
            samples.push((input, random_word, 0)); // Negative sample
        }
    }
    samples
}


#[pyfunction]
pub fn build(documents: Vec<Vec<String>>, vocabulary: Vocab, window: Option<usize>) -> PyResult<Vec<(usize, usize, u8)>> {
    let mut examples = Vec::new();
    documents.par_iter().for_each(|doc| {
        let encoded_doc: Vec<usize> = vocabulary.get_ids(&doc);
        let context_window: u32 = window.unwrap_or(5) as u32;
        for w in encoded_doc.windows(context_window) { // Assuming context window size of 5
            let center = w[(context_window / 2) as u32]; // Center word in the context window
            for (i, &word) in w.iter().enumerate() {
                if i != 2 { // Skip the center word
                    examples.push((center, word, 1)); // Positive example
                    examples.extend(vocabulary.negative_sample(center, vocabulary, 5)); // Negative samples
                }
            }
        }
    });
    Ok(examples)
}


#[pymodule]
fn fastvec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Vocab>()?;
    m.add_function(wrap_pyfunction!(build, m)?)?;
    Ok(())
}
