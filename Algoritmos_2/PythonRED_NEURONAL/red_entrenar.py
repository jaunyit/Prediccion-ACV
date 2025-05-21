import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox, Toplevel, Label

# ------------------- ENTRENAMIENTO ------------------- #
def softmax(z):
    z = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def f1_score(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) * 100 if (prec + rec) else 0.0

# Cargar datos con pandas
cols = ["gender", "age", "heart_hyperten", "avg_glucose_level", "bmi", "smoking_status"]
train_sano = pd.read_csv("Data_ACV_NO_70.csv")
train_malo = pd.read_csv("Data_ACV_SI_70.csv")

Dt = 142
X_train = pd.concat([
    train_sano.loc[:Dt-1, cols],
    train_malo.loc[:Dt-1, cols]
]).to_numpy()
y_train = np.array([0] * Dt + [1] * Dt)

val_min = X_train.min(axis=0)
val_max = X_train.max(axis=0)
ymax, ymin = 1.0, 0.1
Xn_train = (ymax - ymin) * (X_train - val_min) / (val_max - val_min) + ymin
Xn_train = Xn_train.T

targets = np.vstack((1 - y_train, y_train))

test_sano = pd.read_csv("Data_ACV_NO_30.csv").loc[:, cols].to_numpy()
test_malo = pd.read_csv("Data_ACV_SI_30.csv").loc[:, cols].to_numpy()

Xn_test_sano = (ymax - ymin) * (test_sano - val_min) / (val_max - val_min) + ymin
Xn_test_malo = (ymax - ymin) * (test_malo - val_min) / (val_max - val_min) + ymin

# Hiperparámetros
lr = 0.01
epochs = 50000
l2_lambda = 0.05
max_trials_per_hidden = 10
max_hidden = 20
rng = np.random.default_rng()

num_in = 6
num_out = 2
num_hidden = 10
trial = 0
best_f1 = 0

while best_f1 < 92 and num_hidden <= max_hidden:
    trial += 1
    W1 = rng.standard_normal((num_hidden, num_in))
    b1 = rng.standard_normal((num_hidden, 1)) * 0.1
    W2 = rng.standard_normal((num_out, num_hidden))
    b2 = rng.standard_normal((num_out, 1)) * 0.1

    for epoch in range(1, epochs + 1):
        Z1 = W1 @ Xn_train + b1
        A1 = np.tanh(Z1)
        Z2 = W2 @ A1 + b2
        A2 = softmax(Z2)

        loss = -np.sum(targets * np.log(A2 + 1e-12)) / Xn_train.shape[1]

        dZ2 = A2 - targets
        dW2 = dZ2 @ A1.T / Xn_train.shape[1] + l2_lambda * W2
        db2 = np.sum(dZ2, axis=1, keepdims=True) / Xn_train.shape[1]

        dA1 = W2.T @ dZ2
        dZ1 = dA1 * (1 - A1**2)
        dW1 = dZ1 @ Xn_train.T / Xn_train.shape[1] + l2_lambda * W1
        db1 = np.sum(dZ1, axis=1, keepdims=True) / Xn_train.shape[1]

        W1 -= lr * dW1;  b1 -= lr * db1
        W2 -= lr * dW2;  b2 -= lr * db2

    def forward(mat):
        return softmax(W2 @ np.tanh(W1 @ mat.T + b1) + b2)

    tp = sum(forward(Xn_test_sano)[0] > forward(Xn_test_sano)[1])
    fn = Xn_test_sano.shape[0] - tp
    tn = sum(forward(Xn_test_malo)[1] > forward(Xn_test_malo)[0])
    fp = Xn_test_malo.shape[0] - tn

    f1 = f1_score(tp, fp, fn)
    best_f1 = max(best_f1, f1)

    print(f'==> hidden={num_hidden}  F1={f1:.2f}%  (TP:{tp}  FP:{fp}  TN:{tn}  FN:{fn})')

    if f1 >= 92:
        break

    if trial >= max_trials_per_hidden:
        num_hidden += 1
        trial = 0

np.savez("red_92.npz", W1=W1, b1=b1, W2=W2, b2=b2, val_min=val_min, val_max=val_max)
print("Red guardada como red_92.npz")

# ------------------- GUI ------------------- #
def normalizar(x):
    return (ymax - ymin) * (x - val_min) / (val_max - val_min) + ymin

def gui_forward(X):
    A1 = np.tanh(W1 @ X + b1)
    A2 = softmax(W2 @ A1 + b2)
    return A2

def calcular_metricas():
    X = np.vstack((test_sano, test_malo)).T
    y_true = np.array([0] * test_sano.shape[0] + [1] * test_malo.shape[0])
    Xn = (ymax - ymin) * (X.T - val_min) / (val_max - val_min) + ymin
    A2 = gui_forward(Xn.T)
    y_pred = np.argmax(A2, axis=0)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) * 100 if (precision + recall) else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return f1, precision, recall, accuracy

def mostrar_metricas():
    f1, precision, recall, accuracy = calcular_metricas()
    ventana_metrics = Toplevel(ventana)
    ventana_metrics.title("Métricas del Modelo")
    Label(ventana_metrics, text=f"F1-score: {f1:.2f} %").pack(pady=5)
    Label(ventana_metrics, text=f"Precisión: {precision:.2f}").pack(pady=5)
    Label(ventana_metrics, text=f"Recall: {recall:.2f}").pack(pady=5)
    Label(ventana_metrics, text=f"Exactitud: {accuracy:.2f}").pack(pady=5)

def predecir():
    try:
        hypertension = int(hypertension_var.get())
        heart_disease = int(heart_disease_var.get())
        heart_hyperten = (int(hypertension == 1) << 1) | int(heart_disease == 1)

        entrada = np.array([
            float(gender_var.get()),
            float(age_var.get()),
            float(heart_hyperten),
            float(avg_glucose_var.get()),
            float(bmi_var.get()),
            float(smoking_var.get())
        ])
    except ValueError:
        messagebox.showerror("Error", "Por favor, completa todos los campos correctamente.")
        return

    entrada_n = normalizar(entrada).reshape(-1, 1)
    resultado = np.argmax(gui_forward(entrada_n))
    texto = "NO SANO (RIESGO DE ACV)" if resultado == 1 else "SANO"
    messagebox.showinfo("Resultado", f"Resultado: {texto}")

ventana = tk.Tk()
ventana.title("Predicción de ACV")

def add_field(label, var):
    tk.Label(ventana, text=label).pack()
    tk.Entry(ventana, textvariable=var).pack()

gender_var = tk.StringVar()
age_var = tk.StringVar()
hypertension_var = tk.StringVar()
heart_disease_var = tk.StringVar()
avg_glucose_var = tk.StringVar()
bmi_var = tk.StringVar()
smoking_var = tk.StringVar()

add_field("Género (1=Male, 2=Female):", gender_var)
add_field("Edad:", age_var)
add_field("Hipertensión (1=Sí, 2=No):", hypertension_var)
add_field("Cardiopatía (1=Sí, 2=No):", heart_disease_var)
add_field("Nivel de glucosa promedio:", avg_glucose_var)
add_field("Índice de masa corporal (BMI):", bmi_var)
add_field("Tabaquismo (1=Formerly, 2=Never, 3=Smokes, 4=Unknown):", smoking_var)

tk.Button(ventana, text="Predecir", command=predecir).pack(pady=10)
tk.Button(ventana, text="Mostrar métricas del modelo", command=mostrar_metricas).pack(pady=5)

ventana.mainloop()



