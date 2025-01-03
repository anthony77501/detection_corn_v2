from flask import Flask, request, jsonify, render_template
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from torchvision.models import mobilenet_v2
import signal
import sys
from threading import Timer

# Recréer l'architecture du modèle
model = mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, 2)  # 2 classes : mûr et non mûr

# Charger les poids sauvegardés
model.load_state_dict(torch.load('modele_de_mais.pth', map_location=torch.device('cpu')))
model.eval()  # Passer le modèle en mode évaluation

# Transformation des images
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Taille adaptée au modèle
    transforms.ToTensor(),
])

# Application Flask
app = Flask(__name__)

# Arrêter Flask automatiquement après un délai (en secondes) sans activité
TIMEOUT = 300  # Par exemple, 300 secondes = 5 minutes

def shutdown_server():
    print("Arrêt automatique du serveur Flask après inactivité...")
    sys.exit(0)

# Déclencher l'arrêt automatique après TIMEOUT secondes
t = Timer(TIMEOUT, shutdown_server)
t.start()

# Réinitialiser le timer à chaque requête
@app.before_request
def reset_timer():
    global t
    t.cancel()
    t = Timer(TIMEOUT, shutdown_server)
    t.start()

# Interruption manuelle avec Ctrl+C
def signal_handler(signal, frame):
    print("\nArrêt du serveur Flask...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    app.run(debug=True)










@app.route('/')
def home():
    return render_template('index.html')  # Assurez-vous d'avoir un fichier index.html dans 'templates/'

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier n\'a été téléchargé'}), 400

    file = request.files['file']

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

if __name__ == '__main__':
    app.run(debug=True)
