from gerar_gabaritos.id_para_escola import ID_PARA_ESCOLA
from pathlib import Path
import img2pdf
import os

from PyQt6.QtWidgets import QProgressDialog, QApplication
from PyQt6.QtCore import Qt

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

from utils import run_with_progress

def gerar_gabaritos(parent=None):
    Path("gabaritos_pdf").mkdir(parents=True, exist_ok=True)

    total = len(ID_PARA_ESCOLA)

    def worker(cb, cancelled):
        for i, nome_escola in enumerate(ID_PARA_ESCOLA.values(), start=1):
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