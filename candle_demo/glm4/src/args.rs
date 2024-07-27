use clap::Parser;
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(name = "cache", short, long, default_value = ".")]
    pub cache_path: String,

    #[arg(long)]
    pub cpu: bool,

    /// Display the token for the specified prompt.
    #[arg(long)]
    pub verbose_prompt: bool,

    /// The temperature used to generate samples.
    #[arg(long)]
    pub temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    pub top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long)]
    pub seed: Option<u64>,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 5000)]
    pub sample_len: usize,

    #[arg(long)]
    pub model_id: Option<String>,

    #[arg(long)]
    pub revision: Option<String>,

    #[arg(long)]
    pub weight_file: Option<String>,

    #[arg(long)]
    pub tokenizer: Option<String>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    pub repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    pub repeat_last_n: usize,
}
