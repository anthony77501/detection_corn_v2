from flask import Flask, request, jsonify, render_template
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from torchvision.models import mobilenet_v2
import os

# Recréer l'architecture du modèle
model = mobilenet_v2(weights=None)  # Pas de poids pré-entraînés
model.classifier[1] = nn.Linear(model.last_channel, 2)  # 2 classes : mûr et non mûr

# Charger les poids sauvegardés
try:
    model.load_state_dict(torch.load('modele_de_mais.pth', map_location=torch.device('cpu')))
    model.eval()  # Passer le modèle en mode évaluation
    print("Modèle chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {str(e)}")

# Transformation des images
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Taille adaptée au modèle
    transforms.ToTensor(),
])

# Application Flask
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # Limiter la taille des fichiers à 5 MB

# Vérifier les extensions autorisées
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route principale
@app.route('/')
def home():
    return render_template('index.html')  # Assurez-vous d'avoir un fichier index.html dans 'templates/'

# Route pour l'upload et la prédiction
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier n\'a été téléchargé'}), 400

    file = request.files['file']

    if not allowed_file(file.filename):
        return jsonify({'error': 'Format de fichier non supporté'}), 400

    try:
        # Charger et transformer l'image
        image = Image.open(file).convert('RGB')
        image = transform(image).unsqueeze(0)  # Ajouter une dimension pour le batch

        # Faire une prédiction
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            result = 'Mûr' if predicted.item() == 1 else 'Non mûr'

        # Retourner le résultat
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': f'Erreur lors du traitement de l\'image : {str(e)}'}), 500

# Exécuter l'application sur le port requis par Render
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))  # Render fournit le port via la variable d'environnement PORT
    app.run(host='0.0.0.0', port=port)
