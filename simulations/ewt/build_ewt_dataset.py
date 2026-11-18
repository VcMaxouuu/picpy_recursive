"""
Construit le dataset EWT (Universal Dependencies English Web Treebank) au niveau du mot,
aligné sur la tokenisation d'un modèle Pythia.

Étapes :
  1. télécharge les fichiers CoNLL-U de UD_English-EWT (train, dev, test) dans data/raw/ ;
  2. reconstruit le texte de chaque phrase à partir des formes de surface (SpaceAfter=No) ;
  3. tokenise chaque phrase avec le tokenizer du modèle et associe à chaque mot syntaxique
     le DERNIER token du modèle qui le recouvre (le token où le modèle a vu le mot entier) ;
  4. regroupe les phrases d'un même document en séquences de `ctx_len` tokens au plus.

Sorties (dans data/<model>/) :
  tokens.npy : (n_sequences, ctx_len) int32, complété par le token de padding
  words.csv  : une ligne par mot syntaxique, avec sa position (seq, pos) et ses annotations
               (upos, deprel, Tense, Number, VerbForm, Voice, ...)

Usage, depuis le dossier simulations/ewt :
    python build_ewt_dataset.py --model pythia-70m --ctx_len 256
"""
import os
import argparse
import urllib.request
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
UD_URL = "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/{ref}/en_ewt-ud-{split}.conllu"
SPLITS = {"train": "train", "dev": "dev", "test": "test"}
FEATS_KEEP = ["Tense", "Number", "VerbForm", "Voice", "Mood", "Person", "PronType",
              "Definite", "Degree", "NumType", "Case", "Gender"]


def download(ref, raw_dir):
    os.makedirs(raw_dir, exist_ok=True)
    paths = {}
    for split in SPLITS:
        path = os.path.join(raw_dir, f"en_ewt-ud-{split}.conllu")
        if not os.path.exists(path):
            url = UD_URL.format(ref=ref, split=split)
            print(f"Téléchargement de {url}")
            urllib.request.urlretrieve(url, path)
        paths[split] = path
    return paths


def parse_feats(s):
    if s in ("_", ""):
        return {}
    return dict(kv.split("=", 1) for kv in s.split("|") if "=" in kv)


def parse_conllu(path):
    """Renvoie une liste de phrases ; chaque phrase est un dict (doc_id, sent_id, text, words)."""
    sents, doc_id = [], None
    cur = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("# newdoc id"):
                doc_id = line.split("=", 1)[1].strip()
            elif line.startswith("# sent_id"):
                cur = dict(doc_id=doc_id, sent_id=line.split("=", 1)[1].strip(), text=None,
                           surface=[], words=[])
            elif line.startswith("# text ="):
                cur["text"] = line.split("=", 1)[1].strip()
            elif line == "":
                if cur is not None and cur["words"]:
                    sents.append(cur)
                cur = None
            elif not line.startswith("#"):
                cols = line.split("\t")
                tid = cols[0]
                if "." in tid:                       # noeud vide : ignoré
                    continue
                space_after = "SpaceAfter=No" not in cols[9]
                if "-" in tid:                       # token multi-mots (ex. "didn't" = did + n't)
                    a, b = map(int, tid.split("-"))
                    cur["surface"].append(dict(form=cols[1], ids=list(range(a, b + 1)),
                                               space_after=space_after))
                else:
                    i = int(tid)
                    cur["words"].append(dict(id=i, form=cols[1], lemma=cols[2], upos=cols[3],
                                             xpos=cols[4], feats=cols[5], head=cols[6],
                                             deprel=cols[7], misc=cols[9], space_after=space_after))
    return sents


def surface_units(sent):
    """Unités de surface (avec les tokens multi-mots) dans l'ordre, chacune avec ses mots."""
    mwt = {s["ids"][0]: s for s in sent["surface"]}
    covered = {i for s in sent["surface"] for i in s["ids"]}
    words = {w["id"]: w for w in sent["words"]}
    units, i, max_id = [], 1, max(words)
    while i <= max_id:
        if i in mwt:
            u = mwt[i]
            units.append(dict(form=u["form"], word_ids=u["ids"], space_after=u["space_after"]))
            i = u["ids"][-1] + 1
        else:
            if i in words and i not in covered:
                w = words[i]
                units.append(dict(form=w["form"], word_ids=[i], space_after=w["space_after"]))
            i += 1
    return units


def main(args):
    tok = AutoTokenizer.from_pretrained(f"EleutherAI/{args.model}")
    pad_id = tok.pad_token_id
    if pad_id is None:
        pad_id = tok.convert_tokens_to_ids("<|padding|>")
    pad_id = 1 if pad_id is None else int(pad_id)

    paths = download(args.ud_ref, os.path.join(ROOT, "data", "raw"))
    sequences, rows = [], []
    cur_tokens, cur_doc = [], None
    n_text_mismatch = n_truncated = 0

    def flush():
        nonlocal cur_tokens
        if cur_tokens:
            sequences.append(cur_tokens + [pad_id] * (args.ctx_len - len(cur_tokens)))
        cur_tokens = []

    for split, path in paths.items():
        sents = parse_conllu(path)
        print(f"{split} : {len(sents)} phrases")
        for sent in sents:
            if sent["doc_id"] != cur_doc:             # nouvelle séquence à chaque document
                flush()
                cur_doc = sent["doc_id"]

            # texte reconstruit et position (en caractères) de chaque unité de surface
            units = surface_units(sent)
            text, spans = " ", []                     # espace initial, comme Gurnee et al.
            for u in units:
                start = len(text)
                text += u["form"]
                spans.append((start, len(text)))
                if u["space_after"]:
                    text += " "
            text = text.rstrip()
            if sent["text"] is not None and text.strip() != sent["text"].strip():
                n_text_mismatch += 1

            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            if len(ids) > args.ctx_len:
                n_truncated += 1
                continue
            if len(cur_tokens) + len(ids) > args.ctx_len:
                flush()
            base = len(cur_tokens)
            seq_id = len(sequences)
            cur_tokens.extend(ids)

            word_by_id = {w["id"]: w for w in sent["words"]}
            for u, (s, e) in zip(units, spans):
                hit = [k for k, (a, b) in enumerate(offs) if a < e and b > s]
                if not hit:
                    continue
                for wid in u["word_ids"]:
                    w = word_by_id.get(wid)
                    if w is None:
                        continue
                    f = parse_feats(w["feats"])
                    rows.append(dict(
                        split=split, doc_id=sent["doc_id"], sent_id=sent["sent_id"],
                        seq=seq_id, pos=base + hit[-1], first_pos=base + hit[0], n_sub=len(hit),
                        in_mwt=len(u["word_ids"]) > 1, surface=u["form"],
                        form=w["form"], lemma=w["lemma"], upos=w["upos"], xpos=w["xpos"],
                        deprel=w["deprel"], feats=w["feats"],
                        **{k: f.get(k, "") for k in FEATS_KEEP}))
    flush()

    out = os.path.join(ROOT, "data", args.model)
    os.makedirs(out, exist_ok=True)
    tokens = np.asarray(sequences, dtype=np.int32)
    words = pd.DataFrame(rows)
    words.insert(0, "word_id", np.arange(len(words)))
    np.save(os.path.join(out, "tokens.npy"), tokens)
    words.to_csv(os.path.join(out, "words.csv"), index=False)
    print(f"{tokens.shape[0]} séquences de {args.ctx_len} tokens, {len(words)} mots")
    print(f"phrases trop longues ignorées : {n_truncated} ; textes reconstruits différents de '# text' : {n_text_mismatch}")
    print(words.upos.value_counts().head(10).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="pythia-70m")
    p.add_argument("--ctx_len", type=int, default=256)
    p.add_argument("--ud_ref", default="r2.14", help="version (tag git) de UD_English-EWT")
    main(p.parse_args())
