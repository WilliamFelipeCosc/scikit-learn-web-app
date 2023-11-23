from flask import Flask, render_template, request, json, jsonify
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from flask_bootstrap import Bootstrap5

app = Flask(__name__)
bootstrap = Bootstrap5(app)

def get_data():
    df = pd.read_csv("vehicles.csv", nrows=10000)

    df = df[df['manufacturer'].notna()]
    df = df[df['condition'].notna()]
    df = df[df['price'].notna()]
    df = df[df['cylinders'].notna()]

    enc = OneHotEncoder()
    lb = LabelBinarizer()

    selected_columns = ['condition', 'manufacturer']
    X = enc.fit_transform(df[selected_columns]).toarray()
    y = lb.fit_transform(df['cylinders']) 

    # divisao dos dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    return X_train, X_test, y_train, y_test

def generate_results(y_test, y_pred):

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
 
    # matriz de confusão
    conf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    classes = set(y_test[:0])
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.xlabel('Predicted')
    plt.ylabel('True')

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='white' if conf_matrix[i, j] > conf_matrix.max() / 2 else 'black')

    plt.tight_layout()


    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return accuracy, precision, recall, f1, plot_url

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/knn-result', methods=['POST'])
def knn_test():
    X_train, X_test, y_train, y_test = get_data()
    
    data = request.get_json()

    n_neighbors = int(data['k_neighbors'])
    weights = data['k_weights']
    algorithm = data['k_algorithm']
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
   
    # treinamento do modelo
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    accuracy, precision, recall, f1, plot_url = generate_results(y_test, y_pred)

    return jsonify({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'plot_url': plot_url
    })


@app.route('/mlp-result', methods=['POST'])
def mlp_test():
    X_train, X_test, y_train, y_test = get_data()

    data = request.get_json()
   
    # tamanho da camada oculta do formulário ou um valor padrão (100,)
    mlp_hidden_layer_sizes = tuple(map(int, data['mlp_hidden_layer_sizes'].split(',')))
    activation = data['mlp_activation']
    solver = data['mlp_solver']
    mlp = MLPClassifier(hidden_layer_sizes=mlp_hidden_layer_sizes, activation=activation, solver=solver, max_iter=1000)
  
    # treinamento do modelo
    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)

    accuracy, precision, recall, f1, plot_url = generate_results(y_test, y_pred)
    
    return jsonify({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'plot_url': plot_url
    })

@app.route('/dt-result', methods=['POST'])
def dt_test():
    X_train, X_test, y_train, y_test = get_data()

    data = request.get_json()
   
    criterion = data['dt_criterion'] 
    splitter = data['dt_splitter']
    max_depth = int(data['dt_max_depth']) if data['dt_max_depth'] else None
    dt = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth)

    # treinamento do modelo
    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)

    accuracy, precision, recall, f1, plot_url = generate_results(y_test, y_pred)

    return jsonify({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'plot_url': plot_url
    })

@app.route('/rf-result', methods=['POST'])
def rf_test():
    X_train, X_test, y_train, y_test = get_data()

    data = request.get_json()
   
    n_estimators = int(data['rf_n_estimators']) if data['rf_n_estimators'] else 100 
    criterion = data['rf_criterion']
    max_depth = int(data['rf_max_depth']) if data['rf_max_depth'] else None
    rf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth)

    # treinamento do modelo
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    accuracy, precision, recall, f1, plot_url = generate_results(y_test, y_pred)

    return jsonify({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'plot_url': plot_url
    })

if __name__ == '__main__':
    app.run(port=8000, debug=True)
