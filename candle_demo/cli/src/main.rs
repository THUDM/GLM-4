#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle_core as candle;
use candle_core::DType;
use candle_nn::VarBuilder;
use clap::Parser;
use glm4::args::Args;
use glm4::glm4::*;
use glm4::TextGeneration;
use hf_hub::{Repo, RepoType};
use owo_colors::{self, OwoColorize};
use rand::Rng;
use tokenizers::Tokenizer;

fn main() -> Result<(), ()> {
    let args = Args::parse();
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx().red(),
        candle::utils::with_neon().red(),
        candle::utils::with_simd128().red(),
        candle::utils::with_f16c().red(),
    );
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature.unwrap_or(0.95).red(),
        args.repeat_penalty.red(),
        args.repeat_last_n.red(),
    );

    println!("cache path {}", args.cache_path.blue());

    let mut seed: u64 = 0;
    if let Some(_seed) = args.seed {
        seed = _seed;
    } else {
        let mut rng = rand::thread_rng();
        seed = rng.gen();
    }
    println!("Using Seed {}", seed.red());
    let api = hf_hub::api::sync::ApiBuilder::from_cache(hf_hub::Cache::new(args.cache_path.into()))
        .build()
        .unwrap();

    let model_id = match args.model_id {
        Some(model_id) => model_id.to_string(),
        None => "THUDM/glm4-9b".to_string(),
    };
    let revision = match args.revision {
        Some(rev) => rev.to_string(),
        None => "main".to_string(),
    };
    let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));
    let tokenizer_filename = match args.tokenizer {
        Some(file) => std::path::PathBuf::from(file),
        None => api
            .model("THUDM/glm4-9b".to_string())
            .get("tokenizer.json")
            .unwrap(),
    };
    let filenames = match args.weight_file {
        Some(weight_file) => vec![std::path::PathBuf::from(weight_file)],
        None => {
            candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json").unwrap()
        }
    };
    let tokenizer = Tokenizer::from_file(tokenizer_filename).expect("Tokenizer Error");
    let start = std::time::Instant::now();
    let config = Config::glm4();
    let device = candle_examples::device(args.cpu).unwrap();
    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };
    println!("DType is {:?}", dtype.yellow());
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device).unwrap() };
    let model = Model::new(&config, vb).unwrap();

    println!("模型加载完毕 {:?}", start.elapsed().as_secs().green());

    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        seed,
        args.temperature,
        args.top_p,
        args.repeat_penalty,
        args.repeat_last_n,
        args.verbose_prompt,
        &device,
        dtype,
    );
    pipeline.run(args.sample_len)?;
    Ok(())
}
