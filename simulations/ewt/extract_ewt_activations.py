"""
Extrait les activations des neurones MLP (post-GELU) de chaque couche d'un modèle Pythia,
au niveau du mot : pour chaque mot de words.csv, l'activation au dernier token du mot.

Entrées  : data/<model>/tokens.npy et data/<model>/words.csv (build_ewt_dataset.py)
Sorties  : cached_activations/<model>/ewt/<layer>.npy   (n_mots x d_mlp, float16)
           cached_activations/<model>/ewt/words.csv     (les mots retenus, dans le même ordre)

Les mots retenus : position >= --min_pos dans la séquence (comme Gurnee et al., qui ignorent
les 5 premières positions), hors tokens multi-mots, puis --max_words tirés au hasard.
Taille sur disque : max_words x 2048 x 2 octets par couche, soit ~200 Mo pour 50 000 mots.

Usage, depuis le dossier simulations/ewt :
    python extract_ewt_activations.py --model pythia-70m --max_words 50000
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
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
    print(f"Device : {device}")

    data_dir = os.path.join(ROOT, "data", args.model)
    tokens = torch.from_numpy(np.load(os.path.join(data_dir, "tokens.npy")).astype(np.int64))
    words = pd.read_csv(os.path.join(data_dir, "words.csv"), keep_default_na=False)

    keep = (words.pos >= args.min_pos) & (~words.in_mwt.astype(bool))
    words = words[keep]
    if 0 < args.max_words < len(words):
        words = words.sample(n=args.max_words, random_state=args.seed)
    words = words.sort_values(["seq", "pos"]).reset_index(drop=True)
    print(f"{len(words)} mots retenus, dans {words.seq.nunique()} séquences")

    model = AutoModelForCausalLM.from_pretrained(f"EleutherAI/{args.model}").to(device).eval()
    layers = model.gpt_neox.layers
    n_layers, d_mlp = len(layers), model.config.intermediate_size
    acts = {l: np.zeros((len(words), d_mlp), dtype=np.float16) for l in range(n_layers)}

    store = {}

    def make_hook(l):
        def hook(module, inp, out):
            store[l] = out
        return hook

    handles = [layers[l].mlp.act.register_forward_hook(make_hook(l)) for l in range(n_layers)]

    seqs = words.seq.unique()
    rows_by_seq = words.groupby("seq").indices          # seq -> indices de lignes dans `words`
    pos_all = words.pos.to_numpy()
    try:
        for s in tqdm(range(0, len(seqs), args.batch_size)):
            batch_seqs = seqs[s:s + args.batch_size]
            model(tokens[batch_seqs].to(device))
            b_idx, p_idx, r_idx = [], [], []
            for b, sq in enumerate(batch_seqs):
                r = rows_by_seq[sq]
                b_idx.append(np.full(len(r), b))
                p_idx.append(pos_all[r])
                r_idx.append(r)
            b_idx = torch.as_tensor(np.concatenate(b_idx), device=device)
            p_idx = torch.as_tensor(np.concatenate(p_idx), device=device)
            r_idx = np.concatenate(r_idx)
            for l in range(n_layers):
                acts[l][r_idx] = store[l][b_idx, p_idx].float().cpu().numpy().astype(np.float16)
    finally:
        for h in handles:
            h.remove()

    out = os.path.join(ROOT, "cached_activations", args.model, "ewt")
    os.makedirs(out, exist_ok=True)
    for l in range(n_layers):
        np.save(os.path.join(out, f"{l}.npy"), acts[l])
    words.to_csv(os.path.join(out, "words.csv"), index=False)
    print(f"{n_layers} couches de forme {acts[0].shape} enregistrées dans {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="pythia-70m")
    p.add_argument("--max_words", type=int, default=50000, help="0 = tous les mots")
    p.add_argument("--min_pos", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    main(p.parse_args())
