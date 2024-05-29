import tkinter as tk
import threading
import subprocess
import time

def run_python_file():
    status_label.config(text="")
    threading.Thread(target=start_python_script).start()

def start_python_script():
    subprocess.Popen(["python", "D:/StreamLit/MainCodeBase/Merged_Code.py"])
    time.sleep(5)
    status_label.config(text="Video started")

root = tk.Tk()
root.title("AI-Trainer")

status_label = tk.Label(root, text="")
status_label.pack(pady=10)

python_file_button = tk.Button(root, text="Start", command=run_python_file)
python_file_button.pack(pady=10)

root.mainloop()
