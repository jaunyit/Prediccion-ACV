import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import tkinter as tk
from tkinter import messagebox
from sklearn.metrics import confusion_matrix

# === CARGA Y PREPARACIÓN DE DATOS ===
df1 = pd.read_csv("Data_ACV_NO.csv")  # Ya contiene encabezados
df2 = pd.read_csv("Data_ACV_SI.csv")

df = pd.concat([df1, df2], ignore_index=True)

# Eliminamos 'hypertension' y 'heart_disease' de los datos de entrada
X = df.drop(columns=["stroke", "hypertension", "heart_disease"])
y = df["stroke"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# === MOSTRAR DATOS DE ENTRENAMIENTO EN LA CONSOLA ===
print("=== DATOS DE ENTRENAMIENTO ===")
print(X_train.head())
print("\n=== ETIQUETAS DE ENTRENAMIENTO ===")
print(y_train.head())

# === ENTRENAR MODELO ===
modelo = GaussianNB()
modelo.fit(X_train, y_train)

# === GUARDAR MODELO ===
import joblib
joblib.dump(modelo, "modelo_bayes.pkl")


# === EVALUACIÓN DEL MODELO CON MATRIZ DE CONFUSIÓN ===
y_pred = modelo.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Extraer TP, TN, FP, FN
TN, FP, FN, TP = cm.ravel()

# Calcular métricas
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) != 0 else 0
recall = TP / (TP + FN) if (TP + FN) != 0 else 0
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

# Mostrar métricas
print("\n=== MATRIZ DE CONFUSIÓN ===")
print(cm)
print("\n=== MÉTRICAS CALCULADAS ===")
print(f"Exactitud (Accuracy): {accuracy:.4f}")
print(f"Precisión (Precision): {precision:.4f}")
print(f"Recall (Sensibilidad): {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")

# === INTERFAZ GRÁFICA ===
def predecir():
    try:
        # Leer entradas individuales
        genero = float(entradas[0].get())
        edad = float(entradas[1].get())
        hipertension = float(entradas[2].get())
        cardiopatia = float(entradas[3].get())

        # Conversión combinada
        heart_hyperten = hipertension + cardiopatia / 10.0  # Ej: 1 y 2 → 1.2

        glucosa = float(entradas[4].get())
        bmi = float(entradas[5].get())
        fuma = float(entradas[6].get())

        datos = [genero, edad, heart_hyperten, glucosa, bmi, fuma]
        columnas = ["gender", "age", "heart_hyperten", "avg_glucose_level", "bmi", "smoking_status"]
        entrada_df = pd.DataFrame([datos], columns=columnas)

        resultado = modelo.predict(entrada_df)[0]

        if resultado == 1:
            messagebox.showwarning("Resultado", "⚠️ Riesgo de stroke (NO sano)")
        else:
            messagebox.showinfo("Resultado", "✅ Sin riesgo de stroke (SANO)")
    except ValueError:
        messagebox.showerror("Error", "Por favor, ingresa todos los valores numéricos correctamente.")

# === LANZAR INTERFAZ ===
if __name__ == "__main__":
    ventana = tk.Tk()
    ventana.title("Predicción de Stroke")
    ventana.geometry("320x470")

    labels = [
        "Género (1=Hombre, 2=Mujer):",
        "Edad:",
        "Hipertensión (1=Sí, 2=No):",
        "Cardiopatía (1=Sí, 2=No):",
        "Glucosa promedio:",
        "BMI:",
        "Fuma (1=Formerly, 2=Never, 3=Smokes, 4=Unknown):"
    ]
    entradas = []

    for label in labels:
        tk.Label(ventana, text=label).pack()
        entrada = tk.Entry(ventana)
        entrada.pack()
        entradas.append(entrada)

    tk.Button(ventana, text="Predecir", command=predecir).pack(pady=20)

    ventana.mainloop()




