from importlib.metadata import version

import tiktoken
import torch
from torch._functorch.vmap import out_dims_t

if __name__ == "__main__":
    print(version("tiktoken"))
    # tokenizer = tiktoken.get_encoding("gpt2")

    # text = (
    #     "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    #     "of someunknownPlace."
    # )
    # text = "Akwirw ier"
    # integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    # print(integers)
    # print(tokenizer.decode(integers))

    # with open("the-verdict.txt", "r", encoding="utf-8") as f:
    #     raw_text = f.read()
    # print(f"total number of chars: {len(raw_text)}")
    # enc_text = tokenizer.encode(raw_text)
    # print(len(enc_text))

    vocab_size = 6
    output_dim = 3
    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    print(embedding_layer.weight)

    print(embedding_layer(torch.tensor([3])))
    context_length = 4
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    print(pos_embedding_layer.weight)
    pos_vector = []
    for i in range(context_length):
        pos_vector.extend(pos_embedding_layer(torch.tensor([i])))
    print(pos_vector)
