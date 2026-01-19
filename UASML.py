import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# =====================
# LOAD & TRAIN MODEL
# =====================
def load_and_train():
    global df, model

    df = pd.read_csv("dataset test.csv")

    X = df[[
        "Usia", "Gula Darah", "Tekanan Darah",
        "BMI", "Kolesterol", "Perokok (0=Tidak, 1=Ya)"
    ]].values

    y = ((df["Gula Darah"] >= 140) | (df["BMI"] >= 30)).astype(int)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            max_iter=3000,
            random_state=42
        ))
    ])

    model.fit(X, y)

    df["Risiko Diabetes (%)"] = model.predict_proba(X)[:, 1] * 100


load_and_train()

# =====================
# BMI STATUS
# =====================
def status_bmi(bmi):
    if bmi < 18.5:
        return "KURUS"
    elif bmi < 25:
        return "NORMAL"
    elif bmi < 30:
        return "OVERWEIGHT"
    else:
        return "OBESITAS"

# =====================
# REKOMENDASI
# =====================
def rekomendasi(usia, bmi, risiko):
    teks = (
        f"USIA : {usia} TAHUN\n"
        f"STATUS BMI : {status_bmi(bmi)}\n"
        f"RISIKO DIABETES : {risiko:.2f}%\n\n"
    )

    if risiko < 10:
        teks += "KATEGORI: SANGAT RENDAH\n• Pertahankan gaya hidup sehat"
    elif risiko <= 30:
        teks += "KATEGORI: RENDAH\n• Kurangi konsumsi gula"
    elif risiko <= 60:
        teks += "KATEGORI: SEDANG\n• Kontrol gula darah rutin"
    else:
        teks += "KATEGORI: TINGGI\n• Segera konsultasi ke dokter"

    return teks

# =====================
# HASIL INDIVIDU
# =====================
def tampil_hasil(nama, usia, bmi, risiko):
    win = tk.Toplevel(root)
    win.title("Hasil Analisis Diabetes")
    win.geometry("650x520")

    tk.Label(win, text=f"HASIL ANALISIS: {nama}",
             font=("Arial", 14, "bold")).pack(pady=10)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["Risiko Diabetes"], [risiko])
    ax.set_ylim(0, 100)

    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=10)

    tk.Label(win, text=rekomendasi(usia, bmi, risiko),
             justify="left").pack(padx=15, pady=10)

# =====================
# ANALISIS PASIEN
# =====================
def analisis_nama():
    nama = nama_var.get()
    if nama == "":
        messagebox.showerror("Error", "Pilih nama pasien")
        return

    row = df[df["Nama"] == nama].iloc[0]

    data = np.array([[row["Usia"], row["Gula Darah"],
                      row["Tekanan Darah"], row["BMI"],
                      row["Kolesterol"], row["Perokok (0=Tidak, 1=Ya)"]]])

    risiko = model.predict_proba(data)[0][1] * 100
    tampil_hasil(nama, row["Usia"], row["BMI"], risiko)

# =====================
# GRAFIK SEMUA PASIEN
# =====================
def tampil_semua_chart():
    win = tk.Toplevel(root)
    win.title("Grafik Risiko Diabetes")
    win.geometry("900x550")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(df["Nama"], df["Risiko Diabetes (%)"])
    ax.set_ylim(0, 100)
    ax.set_xticklabels(df["Nama"], rotation=45, ha="right")

    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack()

# =====================
# INPUT DATA BARU
# =====================
def tambah_data():
    try:
        data_baru = {
            "Nama": ent_nama.get(),
            "Usia": int(ent_usia.get()),
            "Gula Darah": float(ent_gula.get()),
            "Tekanan Darah": float(ent_tekanan.get()),
            "BMI": float(ent_bmi.get()),
            "Kolesterol": float(ent_kol.get()),
            "Perokok (0=Tidak, 1=Ya)": int(ent_perokok.get())
        }

        pd.DataFrame([data_baru]).to_csv(
            "dataset dibec.csv", mode="a", header=False, index=False
        )

        load_and_train()
        refresh_dropdown()

        messagebox.showinfo("Sukses", "Data berhasil disimpan & model diperbarui")

    except:
        messagebox.showerror("Error", "Input tidak valid")

# =====================
# REFRESH DROPDOWN
# =====================
def refresh_dropdown():
    menu = optionmenu["menu"]
    menu.delete(0, "end")
    for nama in df["Nama"]:
        menu.add_command(label=nama,
                         command=lambda v=nama: nama_var.set(v))

# =====================
# UI UTAMA
# =====================
root = tk.Tk()
root.title("Sistem Prediksi Diabetes")
root.geometry("500x520")

tk.Label(root, text="SISTEM PREDIKSI DIABETES",
         font=("Arial", 14, "bold")).pack(pady=10)

# INPUT FORM
form = tk.Frame(root)
form.pack()

labels = ["Nama", "Usia", "Gula Darah", "Tekanan Darah", "BMI", "Kolesterol", "Perokok (0/1)"]
entries = []

for i, lbl in enumerate(labels):
    tk.Label(form, text=lbl).grid(row=i, column=0, sticky="w")
    ent = tk.Entry(form)
    ent.grid(row=i, column=1)
    entries.append(ent)

(ent_nama, ent_usia, ent_gula,
 ent_tekanan, ent_bmi, ent_kol,
 ent_perokok) = entries

tk.Button(root, text="SIMPAN DATA PASIEN",
          bg="#4CAF50", fg="white",
          command=tambah_data).pack(pady=10)

# ANALISIS
nama_var = tk.StringVar()
optionmenu = tk.OptionMenu(root, nama_var, *df["Nama"])
optionmenu.pack(pady=5)

tk.Button(root, text="ANALISIS PASIEN",
          bg="#2196F3", fg="white",
          command=analisis_nama).pack(pady=5)

tk.Button(root, text="GRAFIK SEMUA PASIEN",
          command=tampil_semua_chart).pack(pady=5)

root.mainloop()
