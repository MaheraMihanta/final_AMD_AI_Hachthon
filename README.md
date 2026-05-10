# Gestion Stock Vision

Prototype cloud-ready pour la gestion de stock medical, la selection d'image,
le futur module de vision par ordinateur et l'ouverture vers un assistant LLM
pour pharmacien.

Le projet ne depend plus de Tkinter pour le chemin principal. Sur un Cloud AMD
headless, il faut lancer l'application web FastAPI.

## Installation Cloud AMD

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` ne contient aucune dependance externe obligatoire. Le
serveur web cloud utilise la bibliotheque standard Python, ce qui evite les
problemes d'installation et les dependances graphiques sur serveur headless.

Il ne contient pas `customtkinter`, car Tkinter/CustomTkinter demande un
environnement graphique et provoque souvent une erreur sur un serveur headless.

## Lancer sur le Cloud AMD

Commande recommandee:

```bash
python start_cloud.py
```

Par defaut, le serveur ecoute sur:

```text
0.0.0.0:8000
```

Si la plateforme fournit un port, utilisez la variable `PORT`:

```bash
PORT=7860 python start_cloud.py
```

Equivalent direct sans lanceur:

```bash
python cloud_app.py
```

`start_cloud.py` configure aussi le module vision par defaut:

- `PILL_DATASET_DIR=./pill_data/pill_yolo`
- `VISION_MODEL_PATH=./models/pill_detector.pt`
- `VISION_POSITIVE_LABELS=Pill Back,Pill Front`

Ces variables peuvent toujours etre remplacees dans l'environnement si le
modele ou le dataset sont stockes ailleurs.

Endpoints utiles:

- `GET /`: interface web headless.
- `GET /health`: verification rapide du serveur.
- `GET /api/inventory`: inventaire en JSON.
- `POST /api/select`: selection article + point vision.
- `POST /api/assistant`: preparation du contexte LLM.

Exemple API:

```bash
curl -X POST http://localhost:8000/api/select \
  -H "Content-Type: application/json" \
  -d '{"query":"MED-001","quantity":2}'
```

## Interface Web

L'interface web dans `cloud_app.py` remplace la GUI Tkinter pour le cloud:

- inventaire charge depuis `data/inventory.csv`;
- formulaire de demande client par nom ou ID;
- selection de l'image associee;
- simulation du deplacement robot;
- zone assistant pharmacien LLM;
- appel explicite au point vision.

Cette interface fonctionne dans un navigateur et ne tente pas d'ouvrir une
fenetre locale.

## Point d'integration prioritaire: vision par ordinateur

Le modele YOLO entraine est branche dans `vision.py`, fonction:

```python
process_image(image_path)
```

Flux actuel:

1. `cloud_app.py` recoit une demande web ou API.
2. `app.py` verifie l'article et le stock avec `prepare_selection(...)`.
3. `cloud_app.py` recupere `article.image_path`.
4. `cloud_app.py` appelle `vision.process_image(image_path)`.
5. Le modele YOLO confirme ou refuse la detection du medicament.
6. Les modules suivants automatisent seulement si la vision confirme.

Les images associees a l'inventaire pointent maintenant vers de vraies images
du dataset `pill_data/pill_yolo/train/images`, afin de tester le modele avec
des exemples issus de l'entrainement. Le verdict `Vision OK` est donne seulement
pour les classes positives configurees, par defaut `Pill Back` et `Pill Front`.

Pour installer les dependances vision:

```bash
python -m pip install -r requirements-vision.txt
```

Tester rapidement une image du dataset:

```bash
python scripts/test_vision_image.py
```

## Evaluation du modele vision

Les resultats d'entrainement YOLO sont dans:

```text
runs/detect/runs/pill_detector/train/results.csv
```

Generer les plots et le resume:

```bash
python scripts/evaluate_pill_detector.py
```

Sorties generees:

- `reports/pill_detector_performance/detection_metrics.png`
- `reports/pill_detector_performance/training_losses.png`
- `reports/pill_detector_performance/validation_losses.png`
- `reports/pill_detector_performance/learning_rates.png`
- `reports/pill_detector_performance/summary.md`

Resume actuel: 50 epochs, meilleur `mAP50=0.96466` a l'epoch 48,
`mAP50-95=0.70187` a l'epoch 50, precision finale `0.94702`,
rappel final `0.93986`.

## Ouverture LLM

Le fichier `llm_assistant.py` prepare un prompt pour un futur assistant
conversationnel. Role prevu: aider le pharmacien a structurer l'echange quand
le client n'a pas d'ordonnance et decrit seulement des symptomes.

Regle importante: le LLM doit assister le pharmacien, pas remplacer son avis.
La recommandation finale et la delivrance restent validees par le pharmacien.

## Simulation Console

Afficher l'inventaire:

```bash
python app.py --list
```

Rechercher un article:

```bash
python app.py --query MED-001 --quantity 2
```

## Option Desktop Locale

La GUI CustomTkinter reste disponible seulement pour un ordinateur avec
environnement graphique:

```bash
python -m pip install -r requirements-desktop.txt
python gui.py
```

Sur le Cloud AMD, utilisez `python start_cloud.py` au lieu de `python gui.py`.

## Tester

```bash
python -m unittest discover -s tests
python -m py_compile app.py cloud_app.py start_cloud.py vision.py pill_dataset.py llm_assistant.py gui.py
```

## Structure

- `cloud_app.py`: application web HTTP standard library pour cloud headless.
- `start_cloud.py`: lanceur serveur avec `HOST` et `PORT`.
- `app.py`: logique de simulation des modules 1 et 2.
- `vision.py`: integration YOLO du module de vision par ordinateur.
- `pill_dataset.py`: detection du dataset YOLO et resolution securisee des images.
- `scripts/evaluate_pill_detector.py`: generation des plots de performance.
- `llm_assistant.py`: preparation du contexte pour un futur LLM pharmacien.
- `gui.py`: interface CustomTkinter optionnelle pour desktop local.
- `data/inventory.csv`: inventaire de depart.
- `images/`: visuels SVG generes pour la demonstration.
- `tests/`: tests des scenarios principaux.
