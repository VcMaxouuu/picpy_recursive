# Neurones de contexte (Europarl, sources de The Pile)

Reprise de la réplication minimale de Gurnee et al. (2023), *Finding Neurons in a Haystack: Case Studies
with Sparse Probing* : https://github.com/wesg52/llm-context-neurons (licence MIT, voir `LICENSE`).

- `activations.py`, `find_context_neurons.py` et `context_neurons/` viennent de ce dépôt (baseline et
  résultats de référence de Gurnee et al.).
- `extract_activations.py` : extraction des activations MLP de Pythia sans TransformerLens.
- `gurnee.ipynb` : boucle de sélection récursive PIC sur les langues (Europarl) et les sources de The Pile,
  comparaison à la baseline, similarité entre classes, ablation et analyse des ordres supérieurs.
- `results/` : supports et métriques enregistrés.

## Données

Les dossiers `data/` et `cached_activations/` ne sont pas versionnés. Pour les reconstruire :

1. copier `data/europarl_lang` et `data/pile_data_source` depuis https://github.com/wesg52/llm-context-neurons ;
2. lancer `python extract_activations.py --model pythia-70m --feature_dataset pile_data_source`
   (et `europarl_lang`).
