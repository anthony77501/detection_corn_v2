from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import onnxruntime as ort
from torchvision import transforms
import os

# Initialisation de Flask
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # Limite à 2 MB

# Charger le modèle ONNX
onnx_model_path = "modele_de_mais.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

# Transformation des images
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Vérifier les extensions autorisées
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')  # Assurez-vous d'avoir un fichier index.html dans 'templates/'

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format'}), 400

    try:
        # Charger et transformer l'image
        image = Image.open(file).convert('RGB')
        image = transform(image).unsqueeze(0).numpy()

        # Exécuter la prédiction ONNX
        inputs = {ort_session.get_inputs()[0].name: image}
        outputs = ort_session.run(None, inputs)
        predicted_class = np.argmax(outputs[0])

        # Retourner le résultat
        result = 'Mûr' if predicted_class == 1 else 'Non mûr'
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': f'Erreur lors du traitement : {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
