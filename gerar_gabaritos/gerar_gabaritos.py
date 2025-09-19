from pathlib import Path
import img2pdf
import os
import sqlite3

from utils import run_with_progress

def files_from_dir(path):
    files = []
    if not os.path.exists(path):
        return files
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isfile(item_path):
            files.append(item_path)
    return files

def gerar_gabaritos(parent=None):
    Path("gabaritos_pdf").mkdir(parents=True, exist_ok=True)

    db = sqlite3.connect("database")
    cursor = db.cursor()
    cursor.execute("SELECT Nome FROM Escola")
    nomes_escola = [linha[0] for linha in cursor.fetchall()]

    total = len(nomes_escola)

    def worker(cb, cancelled):
        for i, nome_escola in enumerate(nomes_escola, start=1):
            if cancelled["flag"]:
                break

            alfa_files = files_from_dir(f"gabaritos/{nome_escola}/alfa")
            beta_files = files_from_dir(f"gabaritos/{nome_escola}/beta")
            extra_alfa_files = files_from_dir(f"gabaritos/{nome_escola}/alfa/extra")
            extra_beta_files = files_from_dir(f"gabaritos/{nome_escola}/beta/extra")

            files = alfa_files + beta_files + extra_alfa_files + extra_beta_files

            if files:    
                with open(f"gabaritos_pdf/gabaritos_{nome_escola}.pdf", "wb") as f:
                    f.write(img2pdf.convert(files))

            cb(i, total)

    run_with_progress(parent, "Gerando gabaritos...", total, worker)