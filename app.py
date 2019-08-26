from flask import Flask, request, redirect, render_template
from markupsafe import Markup

import numpy as np
import cv2 as cv
from main import main

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__, static_url_path='/static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def bytes_to_image(bytes):
    np_arr = np.frombuffer(bytes, np.uint8)
    arr = cv.imdecode(np_arr, cv.IMREAD_COLOR)
    return arr


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            image = bytes_to_image(file.read())

            tables = {}
            for i, table in enumerate(main(image), 0):
                tables[i] = Markup(table.to_html(index=False))

            return render_template('main.html', tables=tables)

    return render_template('main.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0')
