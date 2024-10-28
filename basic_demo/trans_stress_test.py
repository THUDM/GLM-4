"""
Note:
    Using with glm-4-9b-chat-hf will require `transformers>=4.46.0".
"""
import argparse
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
import torch
from threading import Thread

MODEL_PATH = 'THUDM/glm-4-9b-chat-hf'


def stress_test(token_len, n, num_gpu):
    device = torch.device(f"cuda:{num_gpu - 1}" if torch.cuda.is_available() and num_gpu > 0 else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, paddsing_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).to(device).eval()

    # Use INT4 weight infer
    # model = AutoModelForCausalLM.from_pretrained(
    #     MODEL_PATH,
    #     trust_remote_code=True,
    #     quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    #     low_cpu_mem_usage=True,
    # ).eval()

    times = []
    decode_times = []

    print("Warming up...")
    vocab_size = tokenizer.vocab_size
    warmup_token_len = 20
    random_token_ids = torch.randint(3, vocab_size - 200, (warmup_token_len - 5,), dtype=torch.long)
    start_tokens = [151331, 151333, 151336, 198]
    end_tokens = [151337]
    input_ids = torch.tensor(start_tokens + random_token_ids.tolist() + end_tokens, dtype=torch.long).unsqueeze(0).to(
        device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bfloat16).to(device)
    position_ids = torch.arange(len(input_ids[0]), dtype=torch.bfloat16).unsqueeze(0).to(device)
    warmup_inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'position_ids': position_ids
    }
    with torch.no_grad():
        _ = model.generate(
            input_ids=warmup_inputs['input_ids'],
            attention_mask=warmup_inputs['attention_mask'],
            max_new_tokens=2048,
            do_sample=False,
            repetition_penalty=1.0,
            eos_token_id=[151329, 151336, 151338]
        )
    print("Warming up complete. Starting stress test...")

    for i in range(n):
        random_token_ids = torch.randint(3, vocab_size - 200, (token_len - 5,), dtype=torch.long)
        input_ids = torch.tensor(start_tokens + random_token_ids.tolist() + end_tokens, dtype=torch.long).unsqueeze(
            0).to(device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bfloat16).to(device)
        position_ids = torch.arange(len(input_ids[0]), dtype=torch.bfloat16).unsqueeze(0).to(device)
        test_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids
        }

        streamer = TextIteratorStreamer(
            tokenizer=tokenizer,
            timeout=36000,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generate_kwargs = {
            "input_ids": test_inputs['input_ids'],
            "attention_mask": test_inputs['attention_mask'],
            "max_new_tokens": 512,
            "do_sample": False,
            "repetition_penalty": 1.0,
            "eos_token_id": [151329, 151336, 151338],
            "streamer": streamer
        }

        start_time = time.time()
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()

        first_token_time = None
        all_token_times = []

        for token in streamer:
            current_time = time.time()
            if first_token_time is None:
                first_token_time = current_time
                times.append(first_token_time - start_time)
            all_token_times.append(current_time)

        t.join()
        end_time = time.time()

        avg_decode_time_per_token = len(all_token_times) / (end_time - first_token_time) if all_token_times else 0
        decode_times.append(avg_decode_time_per_token)
        print(
            f"Iteration {i + 1}/{n} - Prefilling Time: {times[-1]:.4f} seconds - Average Decode Time: {avg_decode_time_per_token:.4f} tokens/second")

        torch.cuda.empty_cache()

    avg_first_token_time = sum(times) / n
    avg_decode_time = sum(decode_times) / n
    print(f"\nAverage First Token Time over {n} iterations: {avg_first_token_time:.4f} seconds")
    print(f"Average Decode Time per Token over {n} iterations: {avg_decode_time:.4f} tokens/second")
    return times, avg_first_token_time, decode_times, avg_decode_time


def main():
    parser = argparse.ArgumentParser(description="Stress test for model inference")
    parser.add_argument('--token_len', type=int, default=1000, help='Number of tokens for each test')
    parser.add_argument('--n', type=int, default=3, help='Number of iterations for the stress test')
    parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPUs to use for inference')
    args = parser.parse_args()

    token_len = args.token_len
    n = args.n
    num_gpu = args.num_gpu

    stress_test(token_len, n, num_gpu)


if __name__ == "__main__":
    main()
