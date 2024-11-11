from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# تحميل النموذج
model_path = "end3.keras"
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    print("Model file not found.")


# معالجة الصورة بما يناسب النموذج الحالي
def preprocess_image(image):
    image = image.resize((256, 256))  # تغيير حجم الصورة
    image_array = np.array(image) / 255.0  # تطبيع القيم
    return image_array.reshape(1, 256, 256, 3)  # إعادة تشكيل البيانات

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "لم يتم إرسال ملف"}), 400
    
    file = request.files['file']

    if not file.filename.endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({"error": "نوع الملف غير مدعوم. يرجى تحميل صورة."}), 400

    image = Image.open(file.stream)
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    
    # استخراج النتيجة
    class_index = int(np.argmax(predictions, axis=1)[0])
    confidence = float(np.max(predictions))

    # إرسال النتيجة
    response = {
        "class_index": class_index,
        "confidence": confidence
    }
    
    return jsonify(response)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)
