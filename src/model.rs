mod vocab;

pub struct Model {
    pub dim: usize,
    pub negative_sample: usize,
    pub window_size: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub vocab: Option<Vocab>
}