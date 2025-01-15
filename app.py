from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import numpy as np
import onnxruntime as ort
from torchvision import transforms
import os

# Initialisation de Flask
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # Limite à 2 MB
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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

# Texte multilingue
translations = {
    "en": {
        "title": "Corn Ripeness Detection",
        "upload_prompt": "Upload an image to check if the corn is ripe or not.",
        "button_text": "Check",
        "result_ripe": "Ripe",
        "result_not_ripe": "Not Ripe",
        "error_no_file": "No file uploaded.",
        "error_invalid_format": "Invalid file format.",
        "error_processing": "Error during processing.",
        "preview": "Preview the uploaded image below:",
        "confidence": "Confidence:"
    },
    "ja": {
        "title": "トウモロコシの成熟度検出",
        "upload_prompt": "トウモロコシが成熟しているかどうかを確認するには、画像をアップロードしてください。",
        "button_text": "確認",
        "result_ripe": "成熟",
        "result_not_ripe": "未成熟",
        "error_no_file": "ファイルがアップロードされていません。",
        "error_invalid_format": "無効なファイル形式です。",
        "error_processing": "処理中にエラーが発生しました。",
        "preview": "以下にアップロードされた画像をプレビューします：",
        "confidence": "信頼度："
    }
}

@app.route('/')
def home():
    lang = request.args.get('lang', 'en')
    if lang not in translations:
        lang = 'en'
    return render_template('index.html', lang=lang, translations=translations[lang])

@app.route('/upload', methods=['POST'])
def upload():
    lang = request.args.get('lang', 'en')
    if lang not in translations:
        lang = 'en'
    messages = translations[lang]

    if 'file' not in request.files:
        return jsonify({'error': messages['error_no_file']}), 400

    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({'error': messages['error_invalid_format']}), 400

    try:
        # Sauvegarder le fichier pour prévisualisation
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Charger et transformer l'image
        image = Image.open(file_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).numpy()

        # Exécuter la prédiction ONNX
        inputs = {ort_session.get_inputs()[0].name: image_tensor}
        outputs = ort_session.run(None, inputs)
        predicted_class = np.argmax(outputs[0])
        confidence = float(np.max(outputs[0])) * 100  # Calculer la confiance en pourcentage

        # Résultat
        result = messages['result_ripe'] if predicted_class == 1 else messages['result_not_ripe']
        return jsonify({
            'result': result,
            'confidence': f"{messages['confidence']} {confidence:.2f}%",
            'preview_url': f'/uploads/{file.filename}'
        })
    except Exception as e:
        return jsonify({'error': f"{messages['error_processing']}: {str(e)}"}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
