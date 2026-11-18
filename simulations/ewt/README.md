# Simulations EWT : hiérarchie de concepts grammaticaux

Teste si les supports successifs de la boucle de sélection récursive (PIC) passent d'un concept
spécifique (« verbe au passif ») à un concept plus général qui le contient (« verbe »), sur les
activations MLP de Pythia-70m et le corpus Universal Dependencies English Web Treebank (EWT).

## Étapes

Toutes les commandes se lancent depuis ce dossier (`simulations/ewt`).

1. **Construire le dataset** (télécharge UD_English-EWT depuis GitHub, environ 15 Mo, puis aligne
   chaque mot sur les tokens de Pythia) :

   ```bash
   python build_ewt_dataset.py --model pythia-70m --ctx_len 256
   ```

   Sorties : `data/raw/*.conllu`, `data/pythia-70m/tokens.npy`, `data/pythia-70m/words.csv`.

2. **Extraire les activations** au dernier token de chaque mot, pour les 6 couches :

   ```bash
   python extract_ewt_activations.py --model pythia-70m --max_words 50000
   ```

   Sorties : `cached_activations/pythia-70m/ewt/{0..5}.npy` (50 000 × 2048, float16, environ
   200 Mo par couche) et `cached_activations/pythia-70m/ewt/words.csv`.
   `--max_words 0` prend tous les mots (environ 240 000, soit environ 1 Go par couche).

3. **Lancer le notebook** `ewt_hierarchy.ipynb`, cellule par cellule. La cellule de la boucle
   enregistre chaque tâche dans `results/pythia-70m/ewt/` et saute celles déjà calculées : on peut
   l'interrompre et la relancer.

## Réglages principaux (première cellule de code du notebook)

| Paramètre | Défaut | Rôle |
|---|---|---|
| `LAYERS` | `[2, 4]` | couches analysées |
| `MAX_ITER` | `10` | itérations max de la boucle |
| `N_TASK_MAX` | `15000` | taille max du jeu de données d'une tâche |
| `ALPHA`, `N_SIMU` | `0.05`, `2000` | calibration du λ pivotal |

13 tâches × 2 couches = 26 boucles. Les supports, la baseline de Gurnee et al. et les métriques
par niveau sont dans un fichier `.pkl` par tâche et par couche.

## Différences avec Gurnee et al.

- Un token par mot (le dernier), au lieu de tous les tokens du mot.
- Les mots des 5 premières positions de chaque séquence sont ignorés, comme chez eux.
- Pythia actuel (et non la version V0 utilisée dans leur article).
- Les tâches enfant sont définies de deux façons : contre tous les autres mots (*global*) et contre
  les autres membres du parent (*within*, comme leur tâche « voix passive »).
