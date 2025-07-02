use rayon::prelude::*;
use pyo3::prelude::*;
use rand::prelude::*;
mod vocab;
mod embedding;
use vocab::Vocab;
use embedding::Embedding;

fn negative_sample(input: usize, vocabulary: &Vocab, num_samples: usize) -> Vec<(usize, usize, u8)> {
    let mut samples = Vec::new();
    let mut rng = rand::rng();
    for _ in 0..num_samples {
        let random_index = rng.random_range(0..(vocabulary.size));
        if (random_index != input) && vocabulary.valid_ids.contains(&random_index) {
            samples.push((input, random_index, 0)); // Negative sample
        }
    }
    samples
}


#[pyfunction]
pub fn build(documents: Vec<Vec<String>>, vocabulary: &Vocab, window: Option<usize>) -> PyResult<Vec<(usize, usize, u8)>> {
    // let mut examples = Vec::new();
    let examples = documents.par_iter().map(|doc| {
        let encoded_doc: Vec<usize> = vocabulary.get_ids(doc.to_vec()).unwrap_or_else(|_| vec![]);
        let context_window: usize = window.unwrap_or(5);
        let mut examples = Vec::new();
        for w in encoded_doc.windows(context_window) { // Assuming context window size of 5
            let center = w[context_window / 2]; // Center word in the context window
            for (i, &word) in w.iter().enumerate() {
                if i != 2 { // Skip the center word
                    examples.push((center, word, 1)); // Positive example
                    examples.extend(negative_sample(center, vocabulary, 5)); // Negative samples
                }
            }
        }
        examples
    }).flatten().collect::<Vec<_>>();
    Ok(examples)
}


#[pymodule]
fn fastvec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Vocab>()?;
    m.add_class::<Embedding>()?;
    m.add_function(wrap_pyfunction!(build, m)?)?;
    Ok(())
}
