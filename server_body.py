import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import sys
import ast
import csv
import numpy as np
from numpy import asarray
from numpy import savetxt
import model_res
import json

from tensorflow.keras.models import load_model

UPLOAD_FOLDER = './data'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model("./model/model.h5")

if __name__ == "__main__":
    app.run(ssl_context='adhoc')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        print('Got request', file=sys.stderr)
        np_processed_data = np.array(ast.literal_eval(request.args.get('input')))
        processed_data = asarray(np_processed_data)
        savetxt(UPLOAD_FOLDER + '/data1.csv', processed_data, delimiter=',')
        return model_res.run_model(model)

if __name__ == "__main__":
    app.run(ssl_context=('', 'key.pem'))
