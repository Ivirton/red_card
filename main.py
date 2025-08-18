# app_completo.py
import sys
import os
import shutil
import time
import math
import json
import cv2
import numpy as np
import fitz  # PyMuPDF

from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QAction, QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog,
    QMessageBox, QWidget, QVBoxLayout, QLabel, QComboBox,
    QScrollArea, QFormLayout, QDialog, QDialogButtonBox,
    QProgressDialog, QHBoxLayout, QSpinBox, QGroupBox
)

# =============================
# CONFIGURAÇÕES GLOBAIS
# =============================
PASTA_TEMP = "imagens_pdf/temp"
PASTA_RESULT = "imagens_pdf/result"
ARQUIVO_GABARITO = "gabarito.txt"
ARQUIVO_BOLHAS = "bolhas.txt"  # formato: linhas "E,x,y,w,h" e "D,x,y,w,h"

# Área de recorte da região das respostas (ajuste se necessário)
CROP = dict(y_inicio=350, y_fim=700, x_inicio=140, x_fim=540)

# Gabarito padrão (20 questões)
GABARITO_CORRETO = [
    'A', 'C', 'C', 'D', 'C','D', 'A', 'B', 'E', 'C',
    'B', 'C', 'D', 'C', 'D','E', 'C', 'C', 'B', 'A'
]

# Parâmetros de decisão (ajuste se necessário)
FILL_MIN = 0.18      # mínimo absoluto para considerar uma bolha realmente preenchida
DIFF_MIN = 0.12      # diferença entre top e 2º para aceitar top (evita empates/fraco)
ROI_EXPAND = 2       # expandir ROI em px para capturar marcações fora do miolo
MORPH_KERNEL = (3, 3)
DEBUG_LOG = False    # True para ver logs detalhados no terminal

# =============================
# UTILITÁRIOS (I/O & leitura)
# =============================
def imread_unicode(path):
    """Lê imagem mesmo com acentos/Unicode nos caminhos (Windows-friendly)."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return cv2.imread(path)

def salvar_gabarito(gabarito, arquivo=ARQUIVO_GABARITO):
    os.makedirs(os.path.dirname(arquivo) or ".", exist_ok=True)
    with open(arquivo, "w", encoding="utf-8") as f:
        f.write(",".join(gabarito))

def carregar_gabarito(arquivo=ARQUIVO_GABARITO):
    if os.path.exists(arquivo):
        with open(arquivo, "r", encoding="utf-8") as f:
            txt = f.read().strip()
            if txt:
                return txt.split(",")
    # fallback
    return GABARITO_CORRETO.copy()

def default_bolhas_layout():
    """Gera layout padrão (mesmo usado anteriormente). Retorna (esq_list, dir_list)."""
    v_col_esq = [79, 101, 125, 150, 174]
    v_lin_esq = [21, 48, 74, 99, 126, 151, 176, 200, 227, 251]
    tam = 20
    bolhas_esq = [(x, y, tam, tam) for y in v_lin_esq for x in v_col_esq]

    v_col_dir = [262, 286, 311, 336, 360]
    v_lin_dir = [22, 46, 72, 97, 124, 149, 174, 200, 227, 251]
    larguras_dir = {(286, 22): 21, (311, 46): 21, (336, 123): 20, (336, 149): 21, (286, 174): 21, (336, 200): 21}
    bolhas_dir = [(x, y, larguras_dir.get((x, y), 20), 20) for y in v_lin_dir for x in v_col_dir]
    return bolhas_esq, bolhas_dir

def salvar_bolhas(bolhas_esq, bolhas_dir, arquivo=ARQUIVO_BOLHAS):
    os.makedirs(os.path.dirname(arquivo) or ".", exist_ok=True)
    with open(arquivo, "w", encoding="utf-8") as f:
        for (x, y, w, h) in bolhas_esq:
            f.write(f"E,{x},{y},{w},{h}\n")
        for (x, y, w, h) in bolhas_dir:
            f.write(f"D,{x},{y},{w},{h}\n")

def carregar_bolhas(arquivo=ARQUIVO_BOLHAS):
    """Lê o arquivo bolhas.txt no formato E,x,y,w,h / D,x,y,w,h.
       Se faltar ou inválido, retorna layout padrão.
    """
    if not os.path.exists(arquivo):
        return default_bolhas_layout()
    esq, dir_ = [], []
    with open(arquivo, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 5:
                continue
            lado, x, y, w, h = parts
            try:
                tpl = (int(x), int(y), int(w), int(h))
            except ValueError:
                continue
            if lado.upper() == "E":
                esq.append(tpl)
            elif lado.upper() == "D":
                dir_.append(tpl)
    # checa integridade
    if len(esq) != 50 or len(dir_) != 50:
        return default_bolhas_layout()
    return esq, dir_

def cvimg_to_qpixmap(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape
    qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)

# =============================
# DETECÇÃO E DECISÃO (por ROI)
# =============================
def calc_fill_ratio(gray_area, x, y, w, h, expand=ROI_EXPAND):
    """Calcula fill ratio dentro do ROI expandido, aplicando Otsu + morph + máscara circular."""
    H, W = gray_area.shape[:2]
    xa = max(0, int(x - expand))
    ya = max(0, int(y - expand))
    xb = min(W, int(x + w + expand))
    yb = min(H, int(y + h + expand))
    roi = gray_area[ya:yb, xa:xb]
    if roi.size == 0 or roi.shape[0] < 3 or roi.shape[1] < 3:
        return 0.0, 255.0  # nada

    blur = cv2.GaussianBlur(roi, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones(MORPH_KERNEL, np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    rh, rw = th.shape[:2]
    mask = np.zeros((rh, rw), dtype=np.uint8)
    # máscara circular central
    center = (rw // 2, rh // 2)
    radius = max(1, min(rw, rh) // 2 - 1)
    cv2.circle(mask, center, radius, 255, -1)

    masked = cv2.bitwise_and(th, th, mask=mask)
    filled = cv2.countNonZero(masked)
    total_mask = cv2.countNonZero(mask)
    fill_ratio = (filled / total_mask) if total_mask > 0 else 0.0
    mean_gray = float(cv2.mean(roi, mask=mask)[0]) if total_mask > 0 else 255.0
    return float(fill_ratio), mean_gray

# =============================
# PROCESSAMENTO DE UM CARTÃO (IMAGEM) COMPLETO
# =============================
def processar_cartao(caminho_imagem, gabarito, pasta_saida=PASTA_RESULT):
    img = imread_unicode(caminho_imagem)
    if img is None:
        print(f"Erro: não foi possível abrir {caminho_imagem}")
        return None

    # Redimensiona (para manter comportamento consistente)
    ALT_MAX, LARG_MAX = 1000, 800
    alt, larg = img.shape[:2]
    escala = min(ALT_MAX / alt, LARG_MAX / larg, 1.0)
    img_resized = cv2.resize(img, (int(larg * escala), int(alt * escala)))
    imagem_original = img_resized.copy()

    # Verifica recorte
    y0, y1 = CROP["y_inicio"], CROP["y_fim"]
    x0, x1 = CROP["x_inicio"], CROP["x_fim"]
    H, W = img_resized.shape[:2]
    if y1 > H or x1 > W:
        print(f"Aviso: imagem {caminho_imagem} menor que CROP definido -> ignorada")
        return None

    area = img_resized[y0:y1, x0:x1]
    gray_area = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)

    bolhas_esq, bolhas_dir = carregar_bolhas()

    # Calcula fill ratio para todas as bolhas (esq + dir)
    fill_esq = [calc_fill_ratio(gray_area, x, y, w, h) for (x, y, w, h) in bolhas_esq]
    fill_dir = [calc_fill_ratio(gray_area, x, y, w, h) for (x, y, w, h) in bolhas_dir]

    # Para cada grupo de 5 determinamos decisão
    letras = ['A', 'B', 'C', 'D', 'E']
    respostas_esq = []
    logs = []

    # esquerda: 50 bolhas -> 10 grupos (questões 1..10)
    for g in range(0, 50, 5):
        group = fill_esq[g:g+5]  # lista de (fill_ratio, mean_gray)
        vals = [v[0] for v in group]
        pairs = list(zip(letras, vals, [v[1] for v in group]))
        sorted_pairs = sorted(enumerate(vals), key=lambda t: t[1], reverse=True)
        top_idx = sorted_pairs[0][0]
        top_fill = vals[top_idx]
        second_fill = sorted_pairs[1][1] if len(sorted_pairs) > 1 else 0.0

        reason = ""
        decision = None
        if top_fill < FILL_MIN:
            decision = "-"  # branco
            reason = f"top_fill {top_fill:.3f} < FILL_MIN {FILL_MIN}"
        elif (top_fill - second_fill) < DIFF_MIN:
            decision = "*"  # anulada/ambígua
            reason = f"dif {top_fill - second_fill:.3f} < DIFF_MIN {DIFF_MIN}"
        else:
            decision = letras[top_idx]
            reason = f"top {top_fill:.3f} ok (2º {second_fill:.3f})"

        respostas_esq.append(decision)
        if DEBUG_LOG:
            print(f"ESQ group {(g//5)+1}: {[(letras[i], vals[i]) for i in range(5)]} -> {decision} ({reason})")
        logs.append(("esq", (g//5)+1, pairs, decision, reason))

    # direita: 50 bolhas -> 10 grupos (questões 11..20)
    respostas_dir = []
    for g in range(0, 50, 5):
        group = fill_dir[g:g+5]
        vals = [v[0] for v in group]
        pairs = list(zip(letras, vals, [v[1] for v in group]))
        sorted_pairs = sorted(enumerate(vals), key=lambda t: t[1], reverse=True)
        top_idx = sorted_pairs[0][0]
        top_fill = vals[top_idx]
        second_fill = sorted_pairs[1][1] if len(sorted_pairs) > 1 else 0.0

        reason = ""
        decision = None
        if top_fill < FILL_MIN:
            decision = "-"
            reason = f"top_fill {top_fill:.3f} < FILL_MIN {FILL_MIN}"
        elif (top_fill - second_fill) < DIFF_MIN:
            decision = "*"
            reason = f"dif {top_fill - second_fill:.3f} < DIFF_MIN {DIFF_MIN}"
        else:
            decision = letras[top_idx]
            reason = f"top {top_fill:.3f} ok (2º {second_fill:.3f})"

        respostas_dir.append(decision)
        if DEBUG_LOG:
            print(f"DIR group {(g//5)+11}: {[(letras[i], vals[i]) for i in range(5)]} -> {decision} ({reason})")
        logs.append(("dir", (g//5)+11, pairs, decision, reason))

    respostas_final = respostas_esq + respostas_dir  # 20 respostas

    # compara com gabarito e desenha retângulos (verde se correta, vermelho se errada)
    acertos = anuladas = brancas = 0
    imagem_saida = area.copy()
    for i, resposta_aluno in enumerate(respostas_final):
        resposta_correta = gabarito[i] if i < len(gabarito) else None
        if resposta_aluno == "*":
            anuladas += 1
            continue
        elif resposta_aluno == "-":
            brancas += 1
            continue
        elif resposta_aluno == resposta_correta:
            acertos += 1
            cor = (0, 255, 0)
        else:
            cor = (0, 0, 255)

        # encontrar coords do grupo e alternativa
        if i < 10:
            grupo_idx = i
            grupo_coords = bolhas_esq[grupo_idx*5:(grupo_idx+1)*5]
        else:
            grupo_idx = i - 10
            grupo_coords = bolhas_dir[grupo_idx*5:(grupo_idx+1)*5]

        if resposta_aluno in letras:
            idx = letras.index(resposta_aluno)
            if 0 <= idx < len(grupo_coords):
                x, y, w, h = grupo_coords[idx]
                # desenha
                cv2.rectangle(imagem_saida, (x, y), (x + w, y + h), cor, 2)

    # recombina a imagem final com a área corrigida
    imagem_final = imagem_original.copy()
    imagem_final[y0:y1, x0:x1] = imagem_saida

    # escreve resumo no rodapé
    resumo = [f"Acertos: {acertos}/{len(gabarito)}", f"Anuladas: {anuladas}", f"Em branco: {brancas}"]
    x_texto, y_texto = 10, imagem_final.shape[0] - 80
    for idx, linha in enumerate(resumo):
        cv2.putText(imagem_final, linha, (x_texto, y_texto + idx*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # salva arquivo final e salva log simples por página
    os.makedirs(pasta_saida, exist_ok=True)
    base, _ = os.path.splitext(os.path.basename(caminho_imagem))
    caminho_saida = os.path.join(pasta_saida, f"{base}_corrigido.png")
    cv2.imwrite(caminho_saida, imagem_final)

    # salva resultado detalhado em txt
    resultado_txt = os.path.join(pasta_saida, f"{base}_resultado.txt")
    with open(resultado_txt, "w", encoding="utf-8") as f:
        f.write(f"Arquivo: {caminho_imagem}\n")
        f.write(f"Resumo: {resumo}\n\n")
        for idx_entry in logs:
            lado, qnum, pairs, decision, reason = idx_entry
            f.write(f"Q{qnum} ({'esq' if lado=='esq' else 'dir'}): decision={decision} reason={reason}\n")
            for alt, fill, mean_g in pairs:
                f.write(f"   {alt}: fill={fill:.3f} mean_gray={mean_g:.1f}\n")
            f.write("\n")

    if DEBUG_LOG:
        print(f"Salvo: {caminho_saida}  (log: {resultado_txt})")
    return caminho_saida

# =============================
# FUNÇÕES PARA PROCESSAMENTO EM LOTE E GERAR PDF FINAL
# =============================
def processar_todas_as_imagens(gabarito, progress_cb=None):
    imagens_corrigidas = []
    os.makedirs(PASTA_RESULT, exist_ok=True)
    arquivos = sorted([a for a in os.listdir(PASTA_TEMP) if a.lower().endswith((".png", ".jpg", ".jpeg"))])
    total = len(arquivos)
    for idx, arquivo in enumerate(arquivos, start=1):
        caminho_img = os.path.join(PASTA_TEMP, arquivo)
        saida = processar_cartao(caminho_img, gabarito, PASTA_RESULT)
        if saida:
            imagens_corrigidas.append(saida)
        if progress_cb:
            progress_cb(idx, total)
    return imagens_corrigidas

def converter_pdf_para_imagens(pdf_path, progress_cb=None):
    os.makedirs(PASTA_TEMP, exist_ok=True)
    pdf = fitz.open(pdf_path)
    total = len(pdf)
    for i in range(total):
        pagina = pdf.load_page(i)
        pix = pagina.get_pixmap(dpi=300)
        img_path = os.path.join(PASTA_TEMP, f"pagina_{i+1}.png")
        pix.save(img_path)
        if progress_cb:
            progress_cb(i+1, total)
    pdf.close()
    return True

def gerar_pdf_final(imagens_corrigidas, caminho_saida_pdf, progress_cb=None):
    doc = fitz.open()
    total = len(imagens_corrigidas)
    for i, img_path in enumerate(imagens_corrigidas, start=1):
        pm = fitz.Pixmap(img_path)
        page = doc.new_page(width=pm.width, height=pm.height)
        page.insert_image(fitz.Rect(0, 0, pm.width, pm.height), pixmap=pm)
        if progress_cb:
            progress_cb(i, total)
    doc.save(caminho_saida_pdf)
    doc.close()

def limpar_pastas():
    if os.path.exists(PASTA_TEMP):
        shutil.rmtree(PASTA_TEMP, ignore_errors=True)
    if os.path.exists(PASTA_RESULT):
        shutil.rmtree(PASTA_RESULT, ignore_errors=True)

# =============================
# EDITOR VISUAL DE BOLHAS
# =============================
class BubbleEditorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reposicionar Bolhas (visual)")
        self.setMinimumSize(900, 700)

        self.bolhas_esq, self.bolhas_dir = carregar_bolhas()
        self.step = 2
        self.preview = None
        self.info_crop_invalido = False

        main_layout = QVBoxLayout(self)
        sel_layout = QHBoxLayout()
        self.lbl_img_path = QLabel("Nenhuma imagem selecionada.")
        btn_sel_img = QPushButton("Escolher Imagem (qualquer pasta)")
        btn_sel_img.clicked.connect(self.escolher_imagem)
        sel_layout.addWidget(self.lbl_img_path)
        sel_layout.addWidget(btn_sel_img)
        main_layout.addLayout(sel_layout)

        self.lbl_preview = QLabel()
        self.lbl_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview.setStyleSheet("background:#eee; border:1px solid #ccc;")
        main_layout.addWidget(self.lbl_preview, stretch=1)

        controls = QHBoxLayout()
        grp_move = QGroupBox("Mover todas as bolhas")
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

        grp_actions = QGroupBox("Ações")
        ac = QHBoxLayout()
        btn_reset = QPushButton("Resetar para padrão")
        btn_reset.clicked.connect(self.reset_layout)
        btn_save = QPushButton("Salvar bolhas.txt")
        btn_save.clicked.connect(self.salvar_layout)
        ac.addWidget(btn_reset)
        ac.addWidget(btn_save)
        grp_actions.setLayout(ac)

        controls.addWidget(grp_move, stretch=1)
        controls.addWidget(grp_actions, stretch=1)
        main_layout.addLayout(controls)

        info = QLabel("Dica: use também as setas do teclado para mover. A visualização usa o recorte definido em CROP.")
        info.setStyleSheet("color:#444;")
        main_layout.addWidget(info)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Left:
            self.move_all(-self.step, 0)
        elif e.key() == Qt.Key.Key_Right:
            self.move_all(self.step, 0)
        elif e.key() == Qt.Key.Key_Up:
            self.move_all(0, -self.step)
        elif e.key() == Qt.Key.Key_Down:
            self.move_all(0, self.step)
        else:
            super().keyPressEvent(e)

    def escolher_imagem(self):
        start_dir = PASTA_TEMP if os.path.exists(PASTA_TEMP) else os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(self, "Escolher imagem", start_dir, "Imagens (*.png *.jpg *.jpeg)")
        if not path:
            return
        self.lbl_img_path.setText(path)

        img = imread_unicode(path)
        if img is None:
            QMessageBox.warning(self, "Erro", "Não foi possível abrir a imagem selecionada.")
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
            self.info_crop_invalido = True
            self.lbl_preview.setPixmap(QPixmap())
            QMessageBox.warning(self, "CROP fora da imagem",
                                "A imagem é menor que a área de recorte definida em CROP.\nUse uma imagem gerada pela conversão do PDF ou ajuste CROP.")
            return

        self.info_crop_invalido = False
        area = img_resized[y0:y1, x0:x1]
        self.preview = area.copy()
        self.draw_overlay()

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
        scaled = pix.scaled(label_w, label_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
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
        salvar_bolhas(self.bolhas_esq, self.bolhas_dir)
        QMessageBox.information(self, "Salvo", f"Coordenadas gravadas em '{ARQUIVO_BOLHAS}'.")

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

# =============================
# MAIN WINDOW (PyQt6)
# =============================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.pdf_path = None
        self.imagens_corrigidas = []
        self.gabarito = carregar_gabarito()
        self.setWindowTitle("Sistema de Correção de Cartões de Resposta")
        self.setGeometry(200, 200, 560, 540)
        layout = QVBoxLayout()
        self.label = QLabel("Escolha uma opção:")
        layout.addWidget(self.label)
        btn1 = QPushButton("1. Selecionar PDF")
        btn1.clicked.connect(self.selecionar_pdf)
        layout.addWidget(btn1)
        btn2 = QPushButton("2. Converter PDF em imagens (TEMP)")
        btn2.clicked.connect(self.converter_pdf)
        layout.addWidget(btn2)
        btn3 = QPushButton("3. Processar imagens (RESULT)")
        btn3.clicked.connect(self.processar_imagens)
        layout.addWidget(btn3)
        btn4 = QPushButton("4. Salvar PDF Final")
        btn4.clicked.connect(self.salvar_pdf_final)
        layout.addWidget(btn4)
        btn5 = QPushButton("5. Editar Gabarito")
        btn5.clicked.connect(self.editar_gabarito)
        layout.addWidget(btn5)
        btn6 = QPushButton("6. Reposicionar Bolhas (visual)")
        btn6.clicked.connect(self.reposicionar_bolhas)
        layout.addWidget(btn6)
        btn7 = QPushButton("7. Limpar TEMP/RESULT")
        btn7.clicked.connect(self.limpar)
        layout.addWidget(btn7)
        btn8 = QPushButton("Sair")
        btn8.clicked.connect(self.close)
        layout.addWidget(btn8)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        menubar = self.menuBar()
        file_menu = menubar.addMenu("Arquivo")
        act_sel = QAction("Selecionar PDF", self)
        act_sel.triggered.connect(self.selecionar_pdf)
        file_menu.addAction(act_sel)

    def _run_with_progress(self, titulo, total_est, worker):
        dlg = QProgressDialog(titulo, "Cancelar", 0, max(1, total_est), self)
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
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
            dlg.setLabelText(f"{titulo}\nProgresso: {atual}/{total}\nTempo decorrido: {int(elapsed)}s | Estimado restante: {int(restante)}s")
            QApplication.processEvents()
            if dlg.wasCanceled():
                cancelled["flag"] = True
        worker(cb, cancelled)
        if cancelled["flag"]:
            QMessageBox.information(self, "Cancelado", "Operação cancelada.")
            return False
        return True

    def selecionar_pdf(self):
        path, _ = QFileDialog.getOpenFileName(self, "Selecionar PDF", "", "PDF (*.pdf)")
        if path:
            self.pdf_path = path
            QMessageBox.information(self, "PDF Selecionado", f"Arquivo:\n{path}")

    def converter_pdf(self):
        if not self.pdf_path:
            QMessageBox.warning(self, "Erro", "Selecione um PDF primeiro.")
            return
        limpar_pastas()
        def worker(cb, cancelled):
            def progress_cb(i, total):
                if cancelled["flag"]:
                    return
                cb(i, total)
            converter_pdf_para_imagens(self.pdf_path, progress_cb)
        self._run_with_progress("Convertendo PDF para imagens (TEMP)...", 1, worker)
        QMessageBox.information(self, "Sucesso", "PDF convertido para imagens.")

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
        self._run_with_progress("Processando imagens (RESULT)...", len(arquivos), worker)
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
        self._run_with_progress("Gerando PDF final...", len(self.imagens_corrigidas), worker)
        QMessageBox.information(self, "PDF Gerado", f"Arquivo salvo em:\n{path}")

    def editar_gabarito(self):
        dialog = GabaritoDialog(self.gabarito, self)
        if dialog.exec():
            self.gabarito = dialog.get_gabarito()
            salvar_gabarito(self.gabarito)
            QMessageBox.information(self, "Gabarito", "Gabarito atualizado e salvo.")

    def reposicionar_bolhas(self):
        dlg = BubbleEditorDialog(self)
        dlg.exec()

    def limpar(self):
        limpar_pastas()
        QMessageBox.information(self, "Limpeza", "TEMP e RESULT foram apagadas!")

# =============================
# MAIN
# =============================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
