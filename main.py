from flask import Flask
from flask import request
import numpy as np
import torch

app = Flask(__name__)


@app.route('/predict')
def predict():
    trestbps = request.args.get("trestbps")
    chol = request.args.get("chol")
    fbs = request.args.get("fbs")
    ca = request.args.get("ca")
    restecg = request.args.get("restecg")
    thalach = request.args.get("thalach")
    cp = request.args.get("cp")
    exang = request.args.get("exang")
    oldpeak = request.args.get("oldpeak")
    slope = request.args.get("slope")
    data = torch.tensor([float(trestbps), float(chol),
                         float(fbs), float(ca),
                         float(restecg), float(thalach),
                         float(cp), float(exang),
                         float(oldpeak), float(slope)]).reshape(-1, 10)
    pred = model(data)
    result = float(np.argmax(pred.detach().numpy()))
    return str(result)


model = torch.load('predict_model')
app.run(port=8085, host="0.0.0.0")
