import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from tkinter import (
    Tk, Label, Entry, Button, DoubleVar, IntVar, StringVar,
    Toplevel, END, OptionMenu, Text
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# --------- ENTRENAMIENTO Y LIMPIEZA ---------
def leer_y_limpiar(ruta_csv: str) -> pd.DataFrame:
    df = pd.read_csv(
        ruta_csv,
        na_values=["_", "NA", "", "--", "NaN"],
        keep_default_na=True
    )
    df.columns = df.columns.str.strip()
    cols_req = [
        "gender", "age", "hypertension", "heart_disease",
        "avg_glucose_level", "bmi", "smoking_status", "stroke"
    ]
    faltantes = [c for c in cols_req if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en el CSV: {faltantes}")
    feature_cols = [c for c in cols_req if c != "stroke"]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    imp = SimpleImputer(strategy="median")
    df[feature_cols] = pd.DataFrame(
        imp.fit_transform(df[feature_cols]),
        index=df.index,
        columns=feature_cols
    )
    df["stroke"] = pd.to_numeric(df["stroke"], errors="coerce")
    df.dropna(subset=["stroke"], inplace=True)
    df["stroke"] = df["stroke"].astype(int)
    return df

def cargar_y_entrenar(train_csv: str):
    df = leer_y_limpiar(train_csv)
    X, y = df.drop(columns=["stroke"]), df["stroke"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_counts = Counter(y)
    strat_arg = y if min(y_counts.values()) >= 2 else None
    X_tr, _, y_tr, _ = train_test_split(
        X_scaled, y,
        test_size=0.3,
        stratify=strat_arg,
        random_state=42
    )
    svm = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
    svm.fit(X_tr, y_tr)
    return svm, scaler

def evaluar_modelo(modelo, scaler, test_csv):
    df = leer_y_limpiar(test_csv)
    X = scaler.transform(df.drop(columns=["stroke"]))
    y_true = df["stroke"]
    y_pred = modelo.predict(X)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    metrics = pd.DataFrame({
        "Métrica": ["Exactitud", "Precisión", "Recall", "F1-Score"],
        "%": [round(accuracy, 2), round(precision, 2), round(recall, 2), round(f1, 2)]
    })
    return metrics, accuracy

# --------- INTERFAZ ---------
class StrokeGUI:
    def __init__(self, master, modelo, scaler):
        self.master = master
        self.modelo = modelo
        self.scaler = scaler
        master.title("Predicción ACV (SVM)")

        self.gender_map = {"Male": 1, "Female": 2}
        self.smoke_map = {
            "formerly smoked": 1,
            "never smoked": 2,
            "smokes": 3,
            "unknown": 4
        }
        self.cond_map = {
            "Sí padece": 1,
            "No padece": 2
        }

        self.gender_var = StringVar(value="Male")
        self.smoke_var = StringVar(value="never smoked")
        self.ht_text = StringVar(value="No padece")
        self.hd_text = StringVar(value="No padece")
        self.age_var = DoubleVar(value=40)
        self.gluc_var = DoubleVar(value=100)
        self.bmi_var = DoubleVar(value=25)

        Label(master, text="Género").grid(row=0, column=0, sticky="e")
        OptionMenu(master, self.gender_var, *self.gender_map.keys()).grid(row=0, column=1)

        Label(master, text="Edad").grid(row=1, column=0, sticky="e")
        Entry(master, textvariable=self.age_var).grid(row=1, column=1)

        Label(master, text="Hipertensión").grid(row=2, column=0, sticky="e")
        OptionMenu(master, self.ht_text, *self.cond_map.keys()).grid(row=2, column=1)

        Label(master, text="Enfermedad cardíaca").grid(row=3, column=0, sticky="e")
        OptionMenu(master, self.hd_text, *self.cond_map.keys()).grid(row=3, column=1)

        Label(master, text="Nivel glucosa promedio").grid(row=4, column=0, sticky="e")
        Entry(master, textvariable=self.gluc_var).grid(row=4, column=1)

        Label(master, text="BMI").grid(row=5, column=0, sticky="e")
        Entry(master, textvariable=self.bmi_var).grid(row=5, column=1)

        Label(master, text="Estado de fumador").grid(row=6, column=0, sticky="e")
        OptionMenu(master, self.smoke_var, *self.smoke_map.keys()).grid(row=6, column=1)

        Button(master, text="Predecir", command=self.predecir).grid(row=7, column=0, columnspan=2, pady=5)
        Button(master, text="Ver reporte completo", command=self.ver_reporte).grid(row=8, column=0, columnspan=2, pady=5)

        self.resultado_label = Label(master, text="", font=("Arial", 12, "bold"))
        self.resultado_label.grid(row=9, column=0, columnspan=2, pady=10)

    def predecir(self):
        gender_code = self.gender_map.get(self.gender_var.get(), 1)
        smoke_code = self.smoke_map.get(self.smoke_var.get(), 4)
        ht_code = self.cond_map.get(self.ht_text.get(), 2)
        hd_code = self.cond_map.get(self.hd_text.get(), 2)

        valores = [
            gender_code,
            self.age_var.get(),
            ht_code,
            hd_code,
            self.gluc_var.get(),
            self.bmi_var.get(),
            smoke_code
        ]
        columnas = ["gender", "age", "hypertension", "heart_disease",
                    "avg_glucose_level", "bmi", "smoking_status"]
        df = pd.DataFrame([valores], columns=columnas)
        df_scaled = self.scaler.transform(df)
        clase = int(self.modelo.predict(df_scaled)[0])
        prob = float(self.modelo.predict_proba(df_scaled)[0][1])
        texto = f"Stroke: {clase}  (probabilidad: {prob:.2%})"
        color = "green" if clase == 0 else "red"
        self.resultado_label.config(text=texto, fg=color)

    def ver_reporte(self):
        rep_df, acc = evaluar_modelo(self.modelo, self.scaler, "Data_ACV_SI_Y_NO_30.csv")
        win = Toplevel(self.master)
        win.title("Reporte de evaluación")
        Label(win, text="Precisión global: {:.2f} %".format(acc), font=("Arial", 11, "bold")).pack(pady=5)
        box = Text(win, width=60, height=10)
        box.insert(END, rep_df.to_string(index=False))
        box.pack()

# --------- MAIN ---------
if __name__ == "__main__":
    modelo, scaler = cargar_y_entrenar("Data_ACV_SI_Y_NO_70.csv")

    # Guardar modelo y scaler
    import joblib
    joblib.dump(modelo, "modelo_svm.pkl")
    joblib.dump(scaler, "scaler_svm.pkl")

    root = Tk()
    app = StrokeGUI(root, modelo, scaler)
    root.mainloop()






