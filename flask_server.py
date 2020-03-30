from flask import Flask, request, jsonify, render_template
import os
import cv2
import base64
from face_landmark_1000.face_service import *
from werkzeug.utils import secure_filename
import time
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
app = Flask('face algo')
face_landmark_handle = FaceLandmarkHandle()
face_detection_handle = FaceDetectionHandle()

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp', 'jpeg'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/face_landmark', methods=['POST', 'GET'])  # 添加路由
def face_landmark():
    if request.method == 'POST':
        f = request.files['file']

        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        user_input = request.form.get("name")

        basepath = os.path.dirname(__file__)  # 当前文件所在路径

        upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))
        f.save(upload_path)

        img = cv2.imread(upload_path)
        result = face_landmark_handle.run(img)
        cv2.imwrite(os.path.join(basepath, 'static/images', 'landmark_result.jpg'), result)
        return render_template('face_landmark_ok.html', userinput=user_input, val1=time.time())

    return render_template('face_landmark.html')


@app.route('/face_detection', methods=['POST', 'GET'])  # 添加路由
def face_detection():
    if request.method == 'POST':
        f = request.files['file']

        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        user_input = request.form.get("name")

        basepath = os.path.dirname(__file__)  # 当前文件所在路径

        upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))
        f.save(upload_path)

        img = cv2.imread(upload_path)
        result = face_detection_handle.run(img)
        cv2.imwrite(os.path.join(basepath, 'static/images', 'detection_result.jpg'), result)
        return render_template('face_detection_ok.html', userinput=user_input, val1=time.time())

    return render_template('face_detection.html')



def base64_cv2(base64_str):
    imgString = base64.b64decode(base64_str)
    nparr = np.fromstring(imgString, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
