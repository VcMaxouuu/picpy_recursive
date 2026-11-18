# picpy

Sélection de variables par le **Pivotal Information Criterion (PIC)** en Python, et expériences de
**sélection récursive** sur les neurones de modèles de langage.

## Le package `picpy`

- `picpy.models` : régressions linéaire et logistique pénalisées (`LinearRegression`, `LogisticRegression`)
- `picpy.penalties` : pénalités L1, SCAD, MCP
- `picpy.calibration` : calibration pivotale de λ (`GaussianCalibrator`, `BinomialCalibrator`)
- `picpy.selection` : `RecursiveSelection` (supports disjoints successifs, arrêt calibré sous le nul)
  et `StabilitySelection`
- `picpy.solvers` : FISTA, sur NumPy ou PyTorch (GPU)

## Simulations

| Dossier | Contenu |
|---|---|
| `simulations/LoopTree/` | Simulation en arbre : la boucle récupère-t-elle une hiérarchie connue ? |
| `simulations/llm-context-neurons/` | Neurones de contexte de Pythia-70m (langues Europarl, sources de The Pile), comparaison à Gurnee et al. (2023) |
| `simulations/ewt/` | Concepts grammaticaux (Universal Dependencies EWT) : hiérarchie parent → enfant, profondeur K, boucle sur toutes les couches |

Chaque dossier de simulation a son README. Les données et les activations ne sont pas versionnées
(plusieurs Go) ; les scripts de chaque dossier les régénèrent. Les résultats (`results/`, fichiers
`.pkl` et `.csv`) sont versionnés.

## Installation

```bash
pip install -r requirements.txt
```

Les notebooks importent `picpy` depuis la racine du dépôt (`sys.path.insert(0, "../..")`).
