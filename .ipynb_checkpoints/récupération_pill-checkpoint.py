import requests
import os
from roboflow import Roboflow

rf = Roboflow(api_key="OFZ9SMXMSS6IwOz2ro7O")
project = rf.workspace().project("pill_data")

# Rechercher toutes les images ayant la classe "pill"
images = project.search(class_name="pill", limit=100)

os.makedirs("pill_data/pill", exist_ok=True)

for img in images:
    # Récupérer l'URL de l'image originale
    url = img['original_url'] # ou via l'API Image Details
    response = requests.get(url)
    with open(f"pill_data/pill/{img['id']}.jpg", "wb") as f:
        f.write(response.content)
