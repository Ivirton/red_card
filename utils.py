import time
from PyQt6.QtWidgets import QProgressDialog, QApplication, QMessageBox
from PyQt6.QtCore import Qt
from pathlib import Path
import shutil

def run_with_progress(parent, titulo, total_est, worker):
    dlg = QProgressDialog(titulo, "Cancelar", 0, max(1, total_est), parent)
    dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
    dlg.setWindowTitle("Ação em Andamento")
    dlg.setMinimumSize(500, 200)
    dlg.resize(600, 300)          
    dlg.setMinimumDuration(0)
    dlg.setValue(0)
    start = time.time()
    cancelled = {"flag": False}

    def cb(atual, total):
        elapsed = time.time() - start
        media = elapsed / max(1, atual)
        restante = media * (total - atual)
        dlg.setMaximum(total)
        dlg.setValue(atual)
        dlg.setLabelText(
            f"{titulo}\nProgresso: {atual}/{total}\n"
            f"Tempo decorrido: {int(elapsed)}s | Estimado restante: {int(restante)}s"
        )
        QApplication.processEvents()
        if dlg.wasCanceled():
            cancelled["flag"] = True

    worker(cb, cancelled)

    if cancelled["flag"]:
        QMessageBox.information(parent, "Cancelado", "Operação cancelada.")
        return False
    return True

def deletar_gabaritos_temp():
    dir_path = Path("gabaritos")
    
    if dir_path.exists() and dir_path.is_dir():
        shutil.rmtree(dir_path)  

def dialogo_confirmacao(parent=None, titulo="Confirmação", mensagem="Deseja continuar?"):
    dlg = QMessageBox(parent)
    dlg.setWindowTitle(titulo)
    dlg.setText(mensagem)
    dlg.setIcon(QMessageBox.Icon.Question)
    
    btn_sim = dlg.addButton("Sim", QMessageBox.ButtonRole.YesRole)
    btn_nao = dlg.addButton("Não", QMessageBox.ButtonRole.NoRole)
    
    dlg.exec()  
    
    return dlg.clickedButton() == btn_sim