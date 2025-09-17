from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor

from PyQt6.QtWidgets import (
    QPushButton, QFileDialog,
    QMessageBox, QWidget, QVBoxLayout, QLabel, QComboBox,
    QScrollArea, QFormLayout, QDialog, QDialogButtonBox,
    QHBoxLayout, QSpinBox, QGroupBox, QInputDialog
)

from utils import *

# =============================
# EDITOR VISUAL DE BOLHAS (por página)
# =============================
class BubbleEditorDialog(QDialog):
    """
    Editor visual para reposicionar bolhas, percorrendo TODAS as imagens da PASTA_TEMP.
    Para cada imagem salva um arquivo específico: <nome>_bolhas.txt
    Inclui botão "Salvar como padrão e aplicar em TODAS".
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reposicionar Bolhas (por página)")
        self.setMinimumSize(1000, 780)

        # Lista de imagens na pasta TEMP
        self.imagens = []
        if os.path.exists(PASTA_TEMP):
            self.imagens = sorted([os.path.join(PASTA_TEMP, a) for a in os.listdir(PASTA_TEMP)
                                   if a.lower().endswith((".png", ".jpg", ".jpeg"))])

        # estado
        self.idx = 0
        self.img_path_atual = None
        self.bolhas_esq, self.bolhas_dir = default_bolhas_layout()
        self.step = 2
        self.preview = None
        self.info_crop_invalido = False

        main_layout = QVBoxLayout(self)

        # Cabeçalho: info + navegação
        header = QHBoxLayout()
        self.lbl_status = QLabel("Sem imagens em TEMP.")
        header.addWidget(self.lbl_status, stretch=1)

        self.btn_prev = QPushButton("⟵ Anterior")
        self.btn_prev.clicked.connect(self.anterior_imagem)
        self.btn_next = QPushButton("Próxima ⟶")
        self.btn_next.clicked.connect(self.proxima_imagem)
        header.addWidget(self.btn_prev)
        header.addWidget(self.btn_next)
        main_layout.addLayout(header)

        # Preview
        self.lbl_preview = QLabel()
        self.lbl_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview.setStyleSheet("background:#eee; border:1px solid #ccc;")
        main_layout.addWidget(self.lbl_preview, stretch=1)

        # Controles
        controls = QHBoxLayout()

        # Grupo mover
        grp_move = QGroupBox("Mover todas as bolhas (página atual)")
        mv = QHBoxLayout()
        self.spin_step = QSpinBox()
        self.spin_step.setRange(1, 50)
        self.spin_step.setValue(self.step)
        self.spin_step.valueChanged.connect(lambda v: setattr(self, "step", v))
        btn_up = QPushButton("▲")
        btn_left = QPushButton("◀")
        btn_right = QPushButton("▶")
        btn_down = QPushButton("▼")
        btn_up.clicked.connect(lambda: self.move_all(0, -self.step))
        btn_down.clicked.connect(lambda: self.move_all(0, self.step))
        btn_left.clicked.connect(lambda: self.move_all(-self.step, 0))
        btn_right.clicked.connect(lambda: self.move_all(self.step, 0))
        mv.addWidget(QLabel("Passo (px):"))
        mv.addWidget(self.spin_step)
        mv.addWidget(btn_left)
        mv.addWidget(btn_up)
        mv.addWidget(btn_down)
        mv.addWidget(btn_right)
        grp_move.setLayout(mv)

        # Grupo ações
        grp_actions = QGroupBox("Ações")
        ac = QHBoxLayout()
        self.btn_reset = QPushButton("Resetar (esta página)")
        self.btn_reset.clicked.connect(self.reset_layout)
        self.btn_save = QPushButton("Salvar (esta página)")
        self.btn_save.clicked.connect(self.salvar_layout)
        self.btn_save_next = QPushButton("Salvar e Próxima")
        self.btn_save_next.clicked.connect(self.salvar_e_proxima)
        # >>> NOVO BOTÃO: Salvar padrão global e aplicar em TODAS as páginas
        self.btn_apply_all = QPushButton("Salvar como padrão e aplicar em TODAS")
        self.btn_apply_all.setStyleSheet("font-weight:bold;")
        self.btn_apply_all.clicked.connect(self.salvar_padrao_e_aplicar_todas)

        ac.addWidget(self.btn_reset)
        ac.addWidget(self.btn_save)
        ac.addWidget(self.btn_save_next)
        ac.addWidget(self.btn_apply_all)
        grp_actions.setLayout(ac)

        controls.addWidget(grp_move, stretch=2)
        controls.addWidget(grp_actions, stretch=4)
        main_layout.addLayout(controls)

        info = QLabel(
            "Dica: use as setas do teclado para mover. A visualização usa o recorte definido em CROP.\n"
            "Cada página possui seu próprio arquivo *_bolhas.txt em imagens_pdf/temp/.\n"
            "O botão “Salvar como padrão e aplicar em TODAS” copia o layout atual para TODAS as páginas e também para o arquivo global bolhas.txt.\n"
            "Ao clicar em “Salvar (esta página)” este layout fica TRAVADO (#LOCK=1) e não será auto-reposicionado no processamento."
        )
        info.setStyleSheet("color:#444;")
        main_layout.addWidget(info)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Inicializa na primeira imagem (se houver)
        if self.imagens:
            self.carregar_imagem_por_indice(0)
        self._atualizar_botoes_navegacao()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Left:
            self.move_all(-self.step, 0)
        elif e.key() == Qt.Key.Key_Right:
            self.move_all(self.step, 0)
        elif e.key() == Qt.Key.Key_Up:
            self.move_all(0, -self.step)
        elif e.key() == Qt.Key.Key_Down:
            self.move_all(0, self.step)
        elif e.key() == Qt.Key.Key_PageDown:
            self.proxima_imagem()
        elif e.key() == Qt.Key.Key_PageUp:
            self.anterior_imagem()
        else:
            super().keyPressEvent(e)

    def _atualizar_botoes_navegacao(self):
        tem_imagens = len(self.imagens) > 0
        self.btn_prev.setEnabled(tem_imagens and self.idx > 0)
        self.btn_next.setEnabled(tem_imagens and self.idx < len(self.imagens) - 1)
        self.btn_save.setEnabled(tem_imagens and self.preview is not None)
        self.btn_save_next.setEnabled(tem_imagens and self.preview is not None)
        self.btn_reset.setEnabled(tem_imagens and self.preview is not None)
        self.btn_apply_all.setEnabled(tem_imagens and self.preview is not None)

    def carregar_imagem_por_indice(self, idx):
        if not (0 <= idx < len(self.imagens)):
            return
        self.idx = idx
        self.img_path_atual = self.imagens[self.idx]

        img = imread_unicode(self.img_path_atual)
        if img is None:
            QMessageBox.warning(self, "Erro", f"Não foi possível abrir a imagem:\n{self.img_path_atual}")
            self.preview = None
            self._atualizar_botoes_navegacao()
            return

        ALT_MAX, LARG_MAX = 1000, 800
        alt, larg = img.shape[:2]
        escala = min(ALT_MAX / alt, LARG_MAX / larg, 1.0)
        img_resized = cv2.resize(img, (int(larg * escala), int(alt * escala)))

        y0, y1 = CROP["y_inicio"], CROP["y_fim"]
        x0, x1 = CROP["x_inicio"], CROP["x_fim"]
        H, W = img_resized.shape[:2]
        if y1 > H or x1 > W:
            self.preview = None
            self.lbl_preview.setPixmap(QPixmap())
            QMessageBox.warning(
                self, "CROP fora da imagem",
                "A imagem é menor que a área de recorte definida em CROP.\n"
                "Use uma imagem gerada pela conversão do PDF ou ajuste CROP."
            )
            self._atualizar_botoes_navegacao()
            return

        area = img_resized[y0:y1, x0:x1]
        self.preview = area.copy()

        # Carrega bolhas da página (ou global/padrão)
        self.bolhas_esq, self.bolhas_dir = carregar_bolhas_da_imagem(self.img_path_atual, usar_fallback_global=True)

        # Status
        arq_pag = caminho_arquivo_bolhas_da_imagem(self.img_path_atual)
        status = f"[{self.idx+1}/{len(self.imagens)}] {os.path.basename(self.img_path_atual)} | "
        if os.path.exists(arq_pag):
            status += f"Usando arquivo específico: {os.path.basename(arq_pag)}"
            if arquivo_tem_lock(arq_pag):
                status += " (LOCK)"
        elif os.path.exists(ARQUIVO_BOLHAS):
            status += f"Sem arquivo específico -> usando global: {ARQUIVO_BOLHAS}"
        else:
            status += "Sem arquivo específico -> LAYOUT PADRÃO"
        self.lbl_status.setText(status)

        self.draw_overlay()
        self._atualizar_botoes_navegacao()

    def draw_overlay(self):
        if self.preview is None:
            return
        canvas = self.preview.copy()
        pix = cvimg_to_qpixmap(canvas)
        painter = QPainter(pix)
        pen = QPen(QColor(0, 120, 255))
        pen.setWidth(2)
        painter.setPen(pen)
        for (x, y, w, h) in self.bolhas_esq:
            painter.drawRect(QRect(int(x), int(y), int(w), int(h)))
        for (x, y, w, h) in self.bolhas_dir:
            painter.drawRect(QRect(int(x), int(y), int(w), int(h)))
        painter.end()
        label_w = max(400, self.lbl_preview.width())
        label_h = max(300, self.lbl_preview.height())
        scaled = pix.scaled(label_w, label_h, Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation)
        self.lbl_preview.setPixmap(scaled)

    def move_all(self, dx, dy):
        if self.preview is None:
            return
        self.bolhas_esq = [(x + dx, y + dy, w, h) for (x, y, w, h) in self.bolhas_esq]
        self.bolhas_dir = [(x + dx, y + dy, w, h) for (x, y, w, h) in self.bolhas_dir]
        self.draw_overlay()

    def reset_layout(self):
        self.bolhas_esq, self.bolhas_dir = default_bolhas_layout()
        self.draw_overlay()

    def salvar_layout(self):
        if not self.img_path_atual:
            return
        arq = salvar_bolhas_da_imagem(self.bolhas_esq, self.bolhas_dir, self.img_path_atual, lock=True)
        QMessageBox.information(self, "Salvo", f"Coordenadas desta página gravadas em:\n{arq}\n(LOCK=1: auto-reposicionamento será ignorado)")

    def salvar_e_proxima(self):
        self.salvar_layout()
        self.proxima_imagem()

    def salvar_padrao_e_aplicar_todas(self):
        """
        Salva o layout atual como padrão GLOBAL (bolhas.txt) e aplica em TODAS as páginas (sobrescreve *_bolhas.txt).
        """
        if not self.imagens:
            QMessageBox.warning(self, "Sem imagens", "Não há imagens em TEMP para aplicar.")
            return
        resp = QMessageBox.question(
            self,
            "Confirmar aplicação para TODAS",
            "Isto vai sobrescrever o layout de TODAS as páginas em imagens_pdf/temp/ "
            "e salvar como padrão global (bolhas.txt).\n\nDeseja continuar?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if resp != QMessageBox.StandardButton.Yes:
            return
        alterados = aplicar_layout_a_todas_as_imagens(self.bolhas_esq, self.bolhas_dir)
        QMessageBox.information(
            self, "Aplicado",
            f"Padrão salvo em '{ARQUIVO_BOLHAS}' e aplicado em {alterados} página(s)."
        )
        # Atualiza preview/status (o atual já está com o mesmo layout)
        self.draw_overlay()

    def proxima_imagem(self):
        if self.idx < len(self.imagens) - 1:
            self.carregar_imagem_por_indice(self.idx + 1)

    def anterior_imagem(self):
        if self.idx > 0:
            self.carregar_imagem_por_indice(self.idx - 1)


# =============================
# DIALOG PARA EDITAR GABARITO
# =============================
class GabaritoDialog(QDialog):
    def __init__(self, gabarito_atual, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Editar Gabarito")
        self.setMinimumSize(400, 500)
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        self.combo_boxes = []
        letras = ["A", "B", "C", "D", "E"]
        for i, resp in enumerate(gabarito_atual, start=1):
            combo = QComboBox()
            combo.addItems(letras)
            if resp in letras:
                combo.setCurrentText(resp)
            form_layout.addRow(f"Questão {i}:", combo)
            self.combo_boxes.append(combo)
        scroll = QScrollArea()
        widget = QWidget()
        widget.setLayout(form_layout)
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        botoes = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        botoes.accepted.connect(self.accept)
        botoes.rejected.connect(self.reject)
        layout.addWidget(botoes)

    def get_gabarito(self):
        return [cb.currentText() for cb in self.combo_boxes]

class TelaCorrecao(QWidget):
    def __init__(self, mainwindow):
        super().__init__()
        self.mainwindow = mainwindow
        self.pdf_path = None
        self.imagens_corrigidas = []
        self.gabarito = carregar_gabarito()

        layout = QVBoxLayout()
        self.label = QLabel("Fluxo de trabalho:")
        layout.addWidget(self.label)

        btn1 = QPushButton("1. Selecionar PDF")
        btn1.clicked.connect(self.selecionar_pdf)
        layout.addWidget(btn1)

        btn2 = QPushButton("2. Converter PDF em imagens (TEMP)")
        btn2.clicked.connect(self.converter_pdf)
        layout.addWidget(btn2)

        btn6 = QPushButton("3. Reposicionar Bolhas (página por página)")
        btn6.clicked.connect(self.reposicionar_bolhas)
        layout.addWidget(btn6)

        btn3 = QPushButton("4. Processar imagens (RESULT)")
        btn3.clicked.connect(self.processar_imagens)
        layout.addWidget(btn3)

        btn4 = QPushButton("5. Salvar PDF Final")
        btn4.clicked.connect(self.salvar_pdf_final)
        layout.addWidget(btn4)

        btn5 = QPushButton("Editar Gabarito")
        btn5.clicked.connect(self.editar_gabarito)
        layout.addWidget(btn5)

        btn7 = QPushButton("Limpar TEMP/RESULT")
        btn7.clicked.connect(self.limpar)
        layout.addWidget(btn7)

        btn8 = QPushButton("Voltar")
        btn8.clicked.connect(self.mainwindow.exibir_tela_inicial)
        layout.addWidget(btn8)

        self.setLayout(layout)
    
    def selecionar_pdf(self):
        path, _ = QFileDialog.getOpenFileName(self, "Selecionar PDF", "", "PDF (*.pdf)")
        if path:
            self.pdf_path = path
            QMessageBox.information(self, "PDF Selecionado", f"Arquivo:\n{path}")

    def converter_pdf(self):
        if not self.pdf_path:
            QMessageBox.warning(self, "Erro", "Selecione um PDF primeiro.")
            return

        cpu = os.cpu_count() or 1
        default_workers = max(1, cpu - 1)
        n, ok = QInputDialog.getInt(
            self, "Número de processos",
            f"Informe o número de processos (recomendado {default_workers}):",
            value=default_workers, min=1, max=max(1, cpu)
        )
        if not ok:
            n = default_workers

        limpar_pastas()

        def worker(cb, cancelled):
            def progress_cb(i, total):
                if cancelled["flag"]:
                    return
                cb(i, total)
            try:
                converter_pdf_para_imagens(self.pdf_path, progress_cb, workers=n, cancelled=cancelled)
            except Exception as e:
                QMessageBox.critical(self, "Erro na conversão", f"Falha ao converter PDF: {e}")

        run_with_progress(self, "Convertendo PDF para imagens (TEMP)...", 1, worker)

        if os.path.exists(PASTA_TEMP):
            qtd_imgs = len([a for a in os.listdir(PASTA_TEMP) if a.lower().endswith((".png", ".jpg", ".jpeg"))])
            QMessageBox.information(
                self, "Sucesso",
                f"PDF convertido para {qtd_imgs} imagens.\n"
                f"Foram gerados arquivos *_bolhas.txt (template) por página em {PASTA_TEMP}."
            )

    def processar_imagens(self):
        if not os.path.exists(PASTA_TEMP):
            QMessageBox.warning(self, "Erro", "Converta o PDF para imagens primeiro.")
            return
        arquivos = sorted([a for a in os.listdir(PASTA_TEMP) if a.lower().endswith((".png", ".jpg", ".jpeg"))])
        if not arquivos:
            QMessageBox.warning(self, "Vazio", "Nenhuma imagem encontrada em TEMP.")
            return
        self.imagens_corrigidas = []

        def worker(cb, cancelled):
            def progress_cb(i, total):
                if cancelled["flag"]:
                    return
                cb(i, total)
            imgs = processar_todas_as_imagens(self.gabarito, progress_cb)
            if not cancelled["flag"]:
                self.imagens_corrigidas = imgs

        run_with_progress(self, "Processando imagens (RESULT)...", len(arquivos), worker)
        if self.imagens_corrigidas:
            QMessageBox.information(self, "OK", f"{len(self.imagens_corrigidas)} imagens processadas.")

    def salvar_pdf_final(self):
        if not self.imagens_corrigidas:
            QMessageBox.warning(self, "Erro", "Processe as imagens primeiro.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Salvar PDF", "", "PDF (*.pdf)")
        if not path:
            return

        def worker(cb, cancelled):
            def progress_cb(i, total):
                if cancelled["flag"]:
                    return
                cb(i, total)
            gerar_pdf_final(self.imagens_corrigidas, path, progress_cb)

            run_with_progress("Gerando PDF final...", len(self.imagens_corrigidas), worker)
        QMessageBox.information(self, "PDF Gerado", f"Arquivo salvo em:\n{path}")

    def editar_gabarito(self):
        dialog = GabaritoDialog(self.gabarito, self)
        if dialog.exec():
            self.gabarito = dialog.get_gabarito()
            salvar_gabarito(self.gabarito)
            QMessageBox.information(self, "Gabarito", "Gabarito atualizado e salvo.")

    def reposicionar_bolhas(self):
        if not os.path.exists(PASTA_TEMP) or not any(
                a.lower().endswith((".png", ".jpg", ".jpeg")) for a in os.listdir(PASTA_TEMP)):
            QMessageBox.warning(self, "Aviso",
                                "Nenhuma imagem em TEMP.\nConverta um PDF antes de reposicionar as bolhas.")
            return
        dlg = BubbleEditorDialog(self)
        dlg.exec()

    def limpar(self):
        limpar_pastas()
        QMessageBox.information(self, "Limpeza", "TEMP e RESULT foram apagadas!")