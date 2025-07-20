import tkinter as tk
from tkinter import messagebox
import joblib

model = joblib.load("spam_model.pkl")
vec = joblib.load("vectorizer.pkl")

def classify():
    msg = entry.get("1.0", tk.END).strip()
    if not msg:
        messagebox.showinfo("Alert", "Please enter a message.")
        return
    data = vec.transform([msg])
    pred = model.predict(data)

    if pred[0] == "spam":
        status.config(text="⚠️ Caution: This looks like a spam message.\nAvoid clicking on any external links!", fg="red")
    else:
        status.config(text="✅ This is a normal message.\nNo need to worry.", fg="green")

win = tk.Tk()
win.title("Mail Classifier")
win.geometry("800x500")

tk.Label(win, text="Paste a mail message:", font=("Arial", 12)).pack(pady=8)
entry = tk.Text(win, height=8, width=60)
entry.pack()

tk.Button(win, text="Check Message", command=classify).pack(pady=10)

status = tk.Label(win, text="", font=("Arial", 12, "bold"))
status.pack()

win.mainloop()
