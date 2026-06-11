import tiktoken
import torch

from config import GPT_CONFIG_124M
from generate import generate_text_simple
from gpt import GPTModel


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


if __name__ == "__main__":
    text = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")

    input_tokens = text_to_token_ids(text, tokenizer)
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()
    output_tokens = generate_text_simple(
        model,
        input_tokens,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"],
    )
    print(token_ids_to_text(output_tokens, tokenizer))
