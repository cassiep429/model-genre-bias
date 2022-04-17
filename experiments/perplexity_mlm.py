import argparse
import json

import datasets
import os
import numpy as np
import transformers
import torch
from tqdm import tqdm

from debias_eval.model import models
from debias_eval.util import generate_experiment_id

parser = argparse.ArgumentParser(
    description="Computes perplexity for MLMs on WikiText-2."
)
parser.add_argument(
    "--model",
    action="store",
    type=str,
    default="BertForMaskedLM",
)
parser.add_argument(
    "--bias_direction",
    action="store",
    type=str,
    help="Path to the file containing the pre-computed bias direction for SentenceDebias.",
)
parser.add_argument(
    "--projection_matrix",
    action="store",
    type=str,
    help="Path to the file containing the pre-computed projection matrix for INLP.",
)
parser.add_argument(
    "--bias_type",
    action="store",
    type=str,
    default="gender",
    help="Determines the prompt used for self-debias.",
)
parser.add_argument(
    "--self_debias",
    action="store_true",
    help="Whether we are evaluating a self-debias model or not.",
)
parser.add_argument(
    "--model_name_or_path",
    type=str,
    help="Path to pretrained model or model identifier from huggingface.co/models.",
    required=True,
)
parser.add_argument(
    "--stride",
    action="store",
    type=int,
    default=256,
    help="Sequence length used for computing perplexity.",
)
parser.add_argument(
    "--load_path",
    action="store",
    type=str,
    default=None,
    help="Path to load debiased model from."
)
parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    help="Directory where all persistent data will be stored.",
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def score(model, tokenizer, tensor_input, self_debias=False):
    repeat_input = tensor_input.repeat(tensor_input.size(-1) - 2, 1)
    mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
    masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
    labels = repeat_input.masked_fill(masked_input != tokenizer.mask_token_id, -100)

    if self_debias:
        # Compute loss per token.
        loss = 0.0
        for input_, label in zip(masked_input, labels):
            with torch.no_grad():
                loss += model.compute_loss_self_debiasing(
                    input_.unsqueeze(0).to(device),
                    label.unsqueeze(0).to(device),
                    debiasing_prefixes=[
                        "The following text discriminates against people because of their gender: "
                    ],
                ).item()

        return loss / (masked_input.size(0) - 2)  # Don't include [CLS] and [SEP].

    else:
        with torch.no_grad():
            loss = model(masked_input.to(device), labels=labels.to(device)).loss
        return loss.item()


if __name__ == "__main__":
    args = parser.parse_args()

    experiment_id = generate_experiment_id(
        name="perplexity",
        model=args.model,
        model_name_or_path=args.model_name_or_path,
    )

    print("Computing perplexity for MLM:")
    print(f" - model: {args.model}")
    print(f" - bias_direction: {args.bias_direction}")
    print(f" - projection_matrix: {args.projection_matrix}")
    print(f" - bias_type: {args.bias_type}")
    print(f" - self_debias: {args.self_debias}")
    print(f" - load_path: {args.load_path}")
    print(f" - stride: {args.stride}")
    print(f" - persistent_dir: {args.persistent_dir}")

    # Load the model.
    kwargs = {}
    if args.bias_direction is not None:
        # Load the pre-computed bias direction for SentenceDebias.
        bias_direction = torch.load(args.bias_direction)
        kwargs["bias_direction"] = bias_direction

    if args.projection_matrix is not None:
        # Load the pre-computed projection matrix for INLP.
        projection_matrix = torch.load(args.projection_matrix)
        kwargs["projection_matrix"] = projection_matrix

    print("=" * 40)
    print(f"Loading: {args.model}")
    model = getattr(models, args.model)(
        args.load_path or args.model_name_or_path, **kwargs
    )
    print("=" * 40)

    # Load tokenizer.
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Potentially move model to GPU.
    if args.self_debias:
        model._model.to(device)
    else:
        model.to(device)

    test = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer(
        "\n\n".join(test["text"]), add_special_tokens=False, return_tensors="pt"
    )

    n_tokens = 0
    losses = []
    n_sequences = encodings.input_ids.size(1) // args.stride
    for i in tqdm(range(n_sequences)):
        offset = i * args.stride
        input_ids = encodings.input_ids[:, offset : offset + args.stride]

        # Increment token count. Don't include [CLS] and [SEP].
        n_tokens += input_ids.size(1)

        # Add [CLS] and [SEP] tokens.
        input_ids = (
            [tokenizer.cls_token_id] + input_ids[0].tolist() + [tokenizer.sep_token_id]
        )
        input_ids = torch.tensor(input_ids, dtype=torch.int64).unsqueeze(0)

        loss = score(model, tokenizer, input_ids, self_debias=args.self_debias)
        loss = loss * (input_ids.size(1) - 2)  # Don't include [CLS] and [SEP].
        losses.append(loss)

        # Compute perplexity.
        ppl = torch.exp(torch.tensor(losses).sum() / n_tokens).item()

        print(f"Perplexity after {n_tokens}: {ppl:.2f}.")

    os.makedirs(f"{args.persistent_dir}/results/perplexity", exist_ok=True)
    with open(
        f"{args.persistent_dir}/results/perplexity/{experiment_id}.json", "w"
    ) as f:
        json.dump(ppl, f)
