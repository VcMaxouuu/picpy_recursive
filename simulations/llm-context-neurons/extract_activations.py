"""
Extrait les activations des neurones MLP (post-GELU, moyenne sur chaque séquence) pour chaque
couche d'un modèle Pythia, sur un dataset tokenisé de llm-context-neurons (ex. data/europarl_lang).
Équivalent de `blocks.{l}.mlp.hook_post` dans TransformerLens, mais sans dépendre de TransformerLens.

Sortie : cached_activations/<model>/<dataset>/<layer>.mean.npy  (n_sequences x d_mlp, float16)
         cached_activations/<model>/<dataset>/labels.npy

Usage :
    python extract_activations.py --model pythia-70m --feature_dataset europarl_lang
    python extract_activations.py --n_max 2000      # test rapide sur un sous-échantillon
"""

import os
import argparse
import numpy as np
import torch
import datasets
from tqdm import tqdm
from transformers import AutoModelForCausalLM

ROOT = os.path.dirname(os.path.abspath(__file__))


def pick_device(requested):
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main(args):
    device = pick_device(args.device)
    torch.set_grad_enabled(False)
    print(f"Device: {device}")

    model = AutoModelForCausalLM.from_pretrained(f"EleutherAI/{args.model}").to(device)
    model.eval()
    layers = model.gpt_neox.layers
    n_layers = len(layers)
    d_mlp = model.config.intermediate_size

    ds = datasets.load_from_disk(os.path.join(ROOT, "data", args.feature_dataset))
    if args.n_max > 0 and args.n_max < len(ds):
        ds = ds.shuffle(seed=0).select(range(args.n_max))
    label_key = "language" if "language" in ds.column_names else "distribution"
    # ds[:] renvoie un dict de colonnes déjà converties, quelle que soit la version de `datasets`
    data = ds.with_format("numpy")[:]
    labels = np.asarray(data[label_key])
    tokens = torch.from_numpy(np.stack(data["tokens"])).long()
    n = tokens.shape[0]

    acts = {l: np.zeros((n, d_mlp), dtype=np.float32) for l in range(n_layers)}
    batch_store = {}

    def make_hook(layer):
        def hook(module, inputs, output):
            # output : (batch, seq, d_mlp) = activations post-GELU -> moyenne sur la séquence
            batch_store[layer] = output.float().mean(dim=1).cpu().numpy()

        return hook

    handles = [
        layers[l].mlp.act.register_forward_hook(make_hook(l)) for l in range(n_layers)
    ]

    for start in tqdm(range(0, n, args.batch_size)):
        batch = tokens[start : start + args.batch_size].to(device)
        model(batch)
        for l in range(n_layers):
            acts[l][start : start + batch.shape[0]] = batch_store[l]

    for h in handles:
        h.remove()

    out = os.path.join(ROOT, args.output_dir, args.model, args.feature_dataset)
    os.makedirs(out, exist_ok=True)
    for l in range(n_layers):
        np.save(os.path.join(out, f"{l}.mean.npy"), acts[l].astype(np.float16))
    np.save(os.path.join(out, "labels.npy"), labels)
    print(f"Saved {n_layers} layers of shape {acts[0].shape} to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="pythia-70m")
    p.add_argument("--feature_dataset", default="pile_data_source")
    p.add_argument("--output_dir", default="cached_activations")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n_max", type=int, default=0, help="0 = tout le dataset")
    p.add_argument("--device", default=None)
    main(p.parse_args())
