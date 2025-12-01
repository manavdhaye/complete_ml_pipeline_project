from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import math
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

app = Flask(__name__)

df = pickle.load(open("notebook\data\df.pkl", "rb"))

@app.route("/")
def index():
    return render_template(
        "index.html",
        company=df["Company"].unique(),
        typename=df["TypeName"].unique(),
        cpu=df["CPU_brand"].unique(),
        gpu=df["Gpu_brand"].unique(),
        os=df["os"].unique()
    )

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Extract data
    company = data["company"]
    typename = data["typename"]
    ram = int(data["ram"])
    weight = float(data["weight"])
    screen_size = float(data["screen_size"])
    touchscreen = 1 if data["touchscreen"] == "yes" else 0
    ips = 1 if data["ips"] == "yes" else 0

    res = data["resolution"]
    x_res = int(res.split("x")[0])
    y_res = int(res.split("x")[1])
    ppi = (((x_res ** 2) + (y_res ** 2)) ** 0.5) / screen_size

    cpu = data["cpu"]
    gpu = data["gpu"]
    os = data["os"]
    hdd = int(data["hdd"])
    ssd = int(data["ssd"])

    custom_data=CustomData(company,typename,ram,weight,touchscreen,ips,ppi,cpu,gpu,os,hdd,ssd)
    custom_data_dataframe=custom_data.get_data_as_dataframe()
    predict_pileline=PredictPipeline()
    print(custom_data_dataframe)
    output=predict_pileline.predict(custom_data_dataframe)
    print(output)

    output_n = math.floor(np.exp(output[0]))
    print(output_n)

    return jsonify({"price": output_n})

if __name__ == "__main__":
    app.run(debug=True)
