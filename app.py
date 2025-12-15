# app.py

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import pdf_logic
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

pdf_logic.initialize_nlp_models()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload_and_process", methods=["POST"])
def upload_and_process():
    
    if 'pdf_file' not in request.files:
        return jsonify({"error": "Không có file nào được chọn"}), 400
    
    file = request.files['pdf_file']
    
    if file.filename == '':
        return jsonify({"error": "Tên file không hợp lệ"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(file_path)
        except Exception as e:
            return jsonify({"error": f"Lỗi lưu file: {e}"}), 500

        result, error = pdf_logic.process_single_article(file_path)
        
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Cảnh báo: Không thể xóa file tạm thời {filename}: {e}")
            
        if error:
            return jsonify({"error": error}), 500
        else:
            return jsonify(result), 200
            
    return jsonify({"error": "Định dạng file không được hỗ trợ (chỉ chấp nhận PDF)"}), 400

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)