pub mod glm4;

pub mod args;

use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use glm4::*;
use owo_colors::{self, OwoColorize};
use std::io::BufRead;
use std::io::BufReader;
use tokenizers::Tokenizer;

pub struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
    verbose_prompt: bool,
    dtype: DType,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        verbose_prompt: bool,
        device: &Device,
        dtype: DType,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer,
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            verbose_prompt,
            device: device.clone(),
            dtype,
        }
    }

    pub fn run(&mut self, sample_len: usize) -> Result<(), ()> {
        use std::io::Write;

        println!("[欢迎使用GLM-4,请输入prompt]");
        let stdin = std::io::stdin();
        let reader = BufReader::new(stdin);
        // 从标准输入读取prompt
        for line in reader.lines() {
            let line = line.expect("Failed to read line");
            let tokens = self.tokenizer.encode(line, true).expect("tokens error");
            if tokens.is_empty() {
                panic!("Empty prompts are not supported in the chatglm model.")
            }
            if self.verbose_prompt {
                for (token, id) in tokens.get_tokens().iter().zip(tokens.get_ids().iter()) {
                    let token = token.replace('▁', " ").replace("<0x0A>", "\n");
                    println!("{id:7} -> '{token}'");
                }
            }
            let eos_token = match self.tokenizer.get_vocab(true).get("<|endoftext|>") {
                Some(token) => *token,
                None => panic!("cannot find the endoftext token"),
            };
            let mut tokens = tokens.get_ids().to_vec();
            let mut generated_tokens = 0usize;

            std::io::stdout().flush().expect("output flush error");
            let start_gen = std::time::Instant::now();

            //            println!("\n 开始生成");
            println!("samplelen {}", sample_len.blue());
            let mut result = vec![];

            for index in 0..sample_len {
                let context_size = if index > 0 { 1 } else { tokens.len() };
                let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
                let input = Tensor::new(ctxt, &self.device)
                    .unwrap()
                    .unsqueeze(0)
                    .expect("create tensor input error");
                let logits = self.model.forward(&input).unwrap();
                let logits = logits.squeeze(0).unwrap().to_dtype(self.dtype).unwrap();
                let logits = if self.repeat_penalty == 1. {
                    logits
                } else {
                    let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                    candle_transformers::utils::apply_repeat_penalty(
                        &logits,
                        self.repeat_penalty,
                        &tokens[start_at..],
                    )
                    .unwrap()
                };

                let next_token = self.logits_processor.sample(&logits).unwrap();
                tokens.push(next_token);
                generated_tokens += 1;
                if next_token == eos_token {
                    break;
                }
                let token = self
                    .tokenizer
                    .decode(&[next_token], true)
                    .expect("Token error");
                if self.verbose_prompt {
                    println!(
                        "[Index: {}] [Raw Token: {}] [Decode Token: {}]",
                        index.blue(),
                        next_token.green(),
                        token.yellow()
                    );
                }
                result.push(token);
                std::io::stdout().flush().unwrap();
            }
            let dt = start_gen.elapsed();
            println!(
                "\n{generated_tokens} tokens generated ({:.2} token/s)",
                generated_tokens as f64 / dt.as_secs_f64(),
            );
            println!("Result:");
            for tokens in result {
                print!("{tokens}");
            }
            self.model.reset_kv_cache(); // 清理模型kv
        }

        Ok(())
    }
}
