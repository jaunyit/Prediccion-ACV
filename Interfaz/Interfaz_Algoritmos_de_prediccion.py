import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd
import joblib
import os

# === CARGA DE MODELOS ===
base_path = os.path.dirname(__file__)  # Ruta del archivo actual
modelo_svm = joblib.load(os.path.join(base_path, "modelo_svm.pkl"))
scaler_svm = joblib.load(os.path.join(base_path, "scaler_svm.pkl"))
modelo_bayes = joblib.load(os.path.join(base_path, "modelo_bayes.pkl"))
red = np.load(os.path.join(base_path, "red_92.npz"))

W1, b1, W2, b2 = red["W1"], red["b1"], red["W2"], red["b2"]
val_min, val_max = red["val_min"], red["val_max"]
ymax, ymin = 1.0, 0.1

# === FUNCIONES DE RED ===
def softmax(z):
    z = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def normalizar_red(x):
    return (ymax - ymin) * (x - val_min) / (val_max - val_min) + ymin

def red_predict(X):
    A1 = np.tanh(W1 @ X + b1)
    A2 = softmax(W2 @ A1 + b2)
    return np.argmax(A2)

# === PREPARAR DATOS DE ENTRADA ===
def obtener_datos():
    try:
        gender = gender_map[genero_var.get()]
        hypertension = cond_map[ht_var.get()]
        heart_disease = cond_map[hd_var.get()]
        smoking = smoke_map[fuma_var.get()]
        age = float(edad_var.get())
        glucose = float(glucosa_var.get())
        bmi = float(bmi_var.get())

        return [gender, age, hypertension, heart_disease, glucose, bmi, smoking]
    except Exception as e:
        messagebox.showerror("Error", f"Entrada inválida: {e}")
        return None


# === PREDICCIÓN INDIVIDUAL ===
def predecir_svm():
    entrada = obtener_datos()
    if entrada is None: return
    df = pd.DataFrame([entrada], columns=cols)
    pred = modelo_svm.predict(scaler_svm.transform(df))[0]
    prob = modelo_svm.predict_proba(scaler_svm.transform(df))[0][1]
    mostrar_resultado("SVM", pred, prob)

def predecir_bayes():
    entrada = obtener_datos()
    if entrada is None: return
    df = pd.DataFrame([entrada], columns=cols)
    pred = modelo_bayes.predict(df)[0]
    prob = modelo_bayes.predict_proba(df)[0][1]
    mostrar_resultado("Bayesiano", pred, prob)

def predecir_red():
    entrada = obtener_datos()
    if entrada is None: return
    X = normalizar_red(np.array(entrada)).reshape(-1, 1)
    pred = red_predict(X)
    mostrar_resultado("Red Neuronal", pred)

# === MOSTRAR RESULTADO ===
def mostrar_resultado(nombre, pred, prob=None):
    texto = "NO SANO ⚠️" if pred else "SANO ✅"
    if prob is not None:
        messagebox.showinfo(nombre, f"Resultado: {texto}\nProbabilidad: {prob:.2%}")
    else:
        messagebox.showinfo(nombre, f"Resultado: {texto}")

# === INTERFAZ ===
ventana = tk.Tk()
ventana.title("Predicción de Stroke")
ventana.geometry("600x670")
ventana.configure(bg="#6facd9")

fuente_label = ("Arial", 14, "bold")
fuente_entry = ("Arial", 13)

# TÍTULO PRINCIPAL
tk.Label(
    ventana,
    text="Predicción de riesgo de ACV",
    font=("Arial", 20, "bold"),
    fg="#212594",
    bg="#6facd9"
).pack(pady=(15, 10))

# INSTRUCCIONES
tk.Label(
    ventana,
    text="Ingresa los siguientes datos para continuar la predicción",
    font=("Arial", 13),
    fg="black",
    bg="#6facd9"
).pack(pady=(0, 15))

cols = ["gender", "age", "hypertension", "heart_disease", "avg_glucose_level", "bmi", "smoking_status"]

# Mapas para desplegables
gender_map = {"Masculino": 1, "Femenino": 2}
cond_map = {"Sí": 1, "No": 2}
smoke_map = {
    "Formerly smoked": 1,
    "Never smoked": 2,
    "Smokes": 3,
    "Unknown": 4
}

# Variables
genero_var = tk.StringVar(value="Masculino")
ht_var = tk.StringVar(value="No")
hd_var = tk.StringVar(value="No")
fuma_var = tk.StringVar(value="Never smoked")
edad_var = tk.StringVar()
glucosa_var = tk.StringVar()
bmi_var = tk.StringVar()

# Campos desplegables
def add_dropdown(label, var, opciones):
    tk.Label(ventana, text=label, font=fuente_label, bg="#6facd9", fg="black").pack(anchor="w", padx=10)
    option_menu = tk.OptionMenu(ventana, var, *opciones)
    option_menu.config(font=("Arial", 13), bg="#aac2f3", fg="black", highlightthickness=0)
    option_menu["menu"].config(font=("Arial", 13))
    option_menu.pack(fill="x", padx=10)

# Entradas manuales
def add_entry(label, var):
    tk.Label(ventana, text=label, font=fuente_label, bg="#6facd9", fg="black").pack(anchor="w", padx=10)
    tk.Entry(ventana, textvariable=var, font=fuente_entry, bg="#aac2f3").pack(fill="x", padx=10)

add_dropdown("Género:", genero_var, gender_map.keys())
add_entry("Edad:", edad_var)
add_dropdown("Hipertensión:", ht_var, cond_map.keys())
add_dropdown("Cardiopatía:", hd_var, cond_map.keys())
add_entry("Nivel de glucosa promedio:", glucosa_var)
add_entry("Índice de masa corporal (BMI):", bmi_var)
add_dropdown("Estado de fumador:", fuma_var, smoke_map.keys())

# Botones
tk.Button(ventana, text="Predecir con SVM", command=predecir_svm, font=fuente_label, bg="white").pack(pady=5)
tk.Button(ventana, text="Predecir con Bayesiano", command=predecir_bayes, font=fuente_label, bg="white").pack(pady=5)
tk.Button(ventana, text="Predecir con Red Neuronal", command=predecir_red, font=fuente_label, bg="white").pack(pady=5)

ventana.mainloop()


