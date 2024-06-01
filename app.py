from flask import Flask,  render_template, request
import pickle
app = Flask(__name__)

with open('iris_model_svm.pkl', 'rb') as file:
    model = pickle.load(file)


@app.route('/',  methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        # Load the trained model

        sepal_length = request.form.get('sepal_length')
        sepal_width = request.form.get('sepal_width')
        petal_length = request.form.get('petal_length')
        petal_width = request.form.get('petal_width')

        # input to model for prediction
        # 2D  array  [[1.0,4.0,3.0,2.0],[2.0,3.0,4.0,1.0]]
        features = [[sepal_length, sepal_width, petal_length, petal_width]]
        # id  0  1 2
        prediction_id = model.predict(features)[0]
        # setosa versicolor  virginica
        # Define the label mapping
        label_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        prediction = label_mapping[prediction_id]
        return render_template('index.html', prediction_text=prediction, features=features)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
# GET, POST


# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     # Load the trained model
#     with open('iris_model.pkl', 'rb') as file:
#         model = pickle.load(file)
#     sepal_length = request.form.get('sepal_length')
#     sepal_width = request.form.get('sepal_width')
#     petal_length = request.form.get('petal_length')
#     petal_width = request.form.get('petal_width')

#     # input to model for prediction
#     # 2D  array  [[1.0,4.0,3.0,2.0],[2.0,3.0,4.0,1.0]]
#     features = [[sepal_length, sepal_width, petal_length, petal_width]]
#     # id  0  1 2
#     prediction_id = model.predict(features)[0]
#     # setosa versicolor  virginica
#     # Define the label mapping
#     label_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
#     prediction = label_mapping[prediction_id]
#     return render_template('index.html', prediction_text=prediction)


# # request: sepal_length, sepal_width, petal_length, petal_width
