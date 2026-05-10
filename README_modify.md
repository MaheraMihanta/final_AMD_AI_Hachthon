# Gestion Stock Vision

Prototype cloud-ready pour la gestion de stock medical, la selection d'image,
l'integration d'un detecteur de pilules et l'ouverture vers un assistant LLM
pour pharmacien.

Le chemin principal fonctionne sur un Cloud AMD headless: aucune fenetre
Tkinter n'est ouverte. L'interface est une page web servie par Python.

## Installation Cloud AMD

Dependances serveur de base:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` ne contient aucune dependance externe obligatoire. Le
serveur web cloud utilise la bibliotheque standard Python.

Dependances vision pour entrainer et executer YOLO:

```bash
python -m pip install -r requirements-vision.txt
```

Dependances du serveur LLM distant:

```bash
python -m pip install -r requirements-llm.txt
```

Sur une image AMD/ROCm, verifiez que PyTorch detecte bien le GPU avant
l'entrainement:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Dataset Pilules

Le code cherche le dataset dans cet ordre:

1. variable `PILL_DATASET_DIR`;
2. `/shared-docker/amd_ai_hackathon/pill_data/pill`;
3. `./pill_data/pill_yolo`;
4. `./pill_fixed`;
5. `./pill_data/pill`.

Si le telechargement Roboflow COCO est incomplet, la structure brute ressemble a:

```text
pill_data/pill/
  README.dataset.txt
  README.roboflow.txt
  train/
  test/
```

Le script attend un format YOLO Roboflow classique:

```text
train/images
train/labels
valid/images
valid/labels
test/images
test/labels
```

Reconstruire ce format a partir des images et annotations disponibles:

```bash
python scripts/build_pill_yolo_dataset.py
```

La sortie par defaut est `pill_data/pill_yolo`. Le constructeur conserve le
split `test` existant, convertit les annotations COCO en labels YOLO et cree un
split `valid` depuis les images `train` disponibles. Pour regenerer le dossier:

```bash
python scripts/build_pill_yolo_dataset.py --overwrite
```

Oui: pour tester le programme complet avec la vision, il faut utiliser des
photos de ce dataset. Les SVG du dossier `images/` servent seulement a simuler
la gestion de stock; un modele YOLO doit recevoir des images raster comme JPG
ou PNG.

## Entrainer le modele de vision

Sur le Cloud AMD:

```bash
python scripts/build_pill_yolo_dataset.py --overwrite
python scripts/train_pill_detector.py --epochs 50 --imgsz 640 --batch 16 --device 0
```

Le script:

1. detecte le dataset;
2. genere `data/generated_pill_dataset.yaml`;
3. lance l'entrainement YOLO a partir de `yolo11n.pt`;
4. copie le meilleur modele dans `models/pill_detector.pt`.

Validation rapide sans entrainement:

```bash
python scripts/train_pill_detector.py --dry-run
```

Tester une image du dataset apres entrainement:

```bash
python scripts/test_vision_image.py
```

Tester une image precise:

```bash
python scripts/test_vision_image.py --image test/images/nom_image.jpg
```

Si vous stockez le modele ailleurs:

```bash
export VISION_MODEL_PATH=/chemin/vers/best.pt
```

## Lancer sur le Cloud AMD

Commande recommandee:

```bash
python start_cloud.py
```

Par defaut, le serveur ecoute sur:

```text
0.0.0.0:8000
```

Si la plateforme fournit un port:

```bash
PORT=7860 python start_cloud.py
```

Endpoints utiles:

- `GET /`: interface web headless.
- `GET /health`: verification rapide du serveur.
- `GET /api/inventory`: inventaire en JSON.
- `GET /api/dataset/samples`: exemples d'images du dataset.
- `POST /api/select`: selection article + appel vision si l'image est compatible.
- `POST /api/vision-test`: inference sur une image du dataset.
- `POST /api/assistant`: envoi du prompt au LLM distant et retour de sa reponse.

Exemple API vision dataset:

```bash
curl -X POST http://localhost:8000/api/vision-test \
  -H "Content-Type: application/json" \
  -d '{"image_path":"test/images/nom_image.jpg"}'
```

## Integration Vision Dans Le Flux

Le point central est dans `vision.py`:

```python
process_image(image_path)
```

Flux actuel:

1. `cloud_app.py` recoit une demande web ou API.
2. `app.py` verifie l'article et le stock avec `prepare_selection(...)`.
3. `cloud_app.py` recupere `article.image_path`.
4. `cloud_app.py` appelle `vision.process_image(image_path)`.
5. Si `models/pill_detector.pt` existe et si l'image est JPG/PNG, YOLO execute l'inference.
6. Le resultat vision revient avec `ok`, `message`, `model_path`, `image_path` et `detections`.

Pour une demonstration vraiment complete, remplacez progressivement les SVG du
CSV par des chemins vers des photos du dataset, ou utilisez la zone "Test
vision dataset" dans l'interface web.

## Integration LLM Distant

Le LLM tourne sur un autre cloud sous forme de serveur API HTTP. Le fichier
`llm_pour_assistance.py` expose ce serveur autour de vLLM:

```bash
python -m pip install -r requirements-llm.txt
PORT=8010 LLM_MODEL=Qwen/Qwen3-4B-Instruct-2507 python llm_pour_assistance.py
```

Endpoint cote LLM:

- `GET /health`: etat du serveur LLM.
- `POST /chat`: recoit `system_prompt`, `user_context` ou `messages`, puis renvoie `answer`.

Si le serveur LLM doit etre protege par un token:

```bash
LLM_API_KEY=secret PORT=8010 python llm_pour_assistance.py
```

Cote application stock/vision, configurez l'URL du LLM avant de lancer le
serveur principal:

```bash
LLM_API_URL=https://votre-cloud-llm.example.com/chat python start_cloud.py
```

Avec token:

```bash
LLM_API_URL=https://votre-cloud-llm.example.com/chat LLM_API_KEY=secret python start_cloud.py
```

Le fichier `llm_assistant.py` construit le contexte pharmacien, appelle l'API
distante avec JSON, gere les erreurs reseau et conserve un fallback lisible si
`LLM_API_URL` n'est pas configure.

Regle importante: le LLM doit assister le pharmacien, pas remplacer son avis.
La recommandation finale et la delivrance restent validees par le pharmacien.

## Simulation Console

```bash
python app.py --list
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
python -m py_compile app.py cloud_app.py start_cloud.py vision.py llm_assistant.py llm_pour_assistance.py pill_dataset.py gui.py scripts/build_pill_yolo_dataset.py scripts/train_pill_detector.py scripts/test_vision_image.py
```

## Structure

- `cloud_app.py`: application web HTTP pour cloud headless.
- `start_cloud.py`: lanceur serveur avec `HOST` et `PORT`.
- `app.py`: logique de simulation des modules 1 et 2.
- `vision.py`: chargement du modele YOLO et inference.
- `pill_dataset.py`: detection du dataset, generation YAML et exemples images.
- `scripts/build_pill_yolo_dataset.py`: reconstruction COCO vers YOLO train/valid/test.
- `scripts/train_pill_detector.py`: entrainement du detecteur.
- `scripts/test_vision_image.py`: test CLI du modele vision.
- `llm_assistant.py`: preparation du contexte et client HTTP vers le LLM distant.
- `llm_pour_assistance.py`: serveur API vLLM a deployer sur le cloud LLM.
- `gui.py`: interface CustomTkinter optionnelle pour desktop local.
- `data/inventory.csv`: inventaire de depart.
- `images/`: visuels SVG pour la demonstration stock.
- `tests/`: tests des scenarios principaux.
