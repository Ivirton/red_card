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
import concurrent.futures

from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QAction, QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog,
    QMessageBox, QWidget, QVBoxLayout, QLabel, QComboBox,
    QScrollArea, QFormLayout, QDialog, QDialogButtonBox,
    QProgressDialog, QHBoxLayout, QSpinBox, QGroupBox, QInputDialog
)

# =============================
# CONFIGURAÇÕES GLOBAIS
# =============================
PASTA_TEMP = "imagens_pdf/temp"
PASTA_RESULT = "imagens_pdf/result"
ARQUIVO_GABARITO = "gabarito.txt"
ARQUIVO_BOLHAS = "bolhas.txt"  # layout global (compatibilidade/backup)

# Área de recorte da região das respostas (ajuste se necessário)
CROP = dict(y_inicio=350, y_fim=700, x_inicio=140, x_fim=540)

# Gabarito padrão (20 questões)
GABARITO_CORRETO = [
    'A', 'C', 'C', 'D', 'C', 'D', 'A', 'B', 'E', 'C',
    'B', 'C', 'D', 'C', 'D', 'E', 'C', 'C', 'B', 'A'
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
    """Salva EM ARQUIVO ESPECÍFICO informado (pode ser o global ou por imagem)."""
    os.makedirs(os.path.dirname(arquivo) or ".", exist_ok=True)
    with open(arquivo, "w", encoding="utf-8") as f:
        for (x, y, w, h) in bolhas_esq:
            f.write(f"E,{x},{y},{w},{h}\n")
        for (x, y, w, h) in bolhas_dir:
            f.write(f"D,{x},{y},{w},{h}\n")


def carregar_bolhas(arquivo=ARQUIVO_BOLHAS):
    """Lê arquivo de bolhas no formato E,x,y,w,h / D,x,y,w,h. Se faltar ou inválido, retorna layout padrão."""
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
    if len(esq) != 50 or len(dir_) != 50:
        return default_bolhas_layout()
    return esq, dir_


def caminho_arquivo_bolhas_da_imagem(img_path):
    """Retorna o caminho do arquivo de bolhas específico para a imagem/página."""
    base = os.path.splitext(os.path.basename(img_path))[0]  # p.ex. 'pagina_1'
    return os.path.join(PASTA_TEMP, f"{base}_bolhas.txt")


def carregar_bolhas_da_imagem(img_path, usar_fallback_global=True):
    """
    Carrega bolhas específicas da imagem. Fallback para ARQUIVO_BOLHAS (global) e, por fim, para o padrão.
    """
    arq_pagina = caminho_arquivo_bolhas_da_imagem(img_path)
    if os.path.exists(arq_pagina):
        return carregar_bolhas(arq_pagina)
    if usar_fallback_global and os.path.exists(ARQUIVO_BOLHAS):
        return carregar_bolhas(ARQUIVO_BOLHAS)
    return default_bolhas_layout()


def salvar_bolhas_da_imagem(bolhas_esq, bolhas_dir, img_path):
    """Salva as bolhas específicas daquela imagem/página."""
    arq_pagina = caminho_arquivo_bolhas_da_imagem(img_path)
    salvar_bolhas(bolhas_esq, bolhas_dir, arq_pagina)
    return arq_pagina


def inicializar_bolhas_para_todas_as_imagens():
    """
    Cria (se não existirem) arquivos _bolhas.txt individuais, com layout padrão,
    para cada imagem presente em PASTA_TEMP. Facilita a edição um-a-um.
    """
    if not os.path.exists(PASTA_TEMP):
        return 0
    arquivos = sorted([a for a in os.listdir(PASTA_TEMP) if a.lower().endswith((".png", ".jpg", ".jpeg"))])
    bolhas_esq_default, bolhas_dir_default = default_bolhas_layout()
    criados = 0
    for nome in arquivos:
        img_path = os.path.join(PASTA_TEMP, nome)
        arq_pag = caminho_arquivo_bolhas_da_imagem(img_path)
        if not os.path.exists(arq_pag):
            salvar_bolhas(bolhas_esq_default, bolhas_dir_default, arq_pag)
            criados += 1
    return criados


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
# RENDERER PARA PROCESSOS (picklable)
# =============================
def _render_page_to_png(task):
    """Worker picklable: task = (pdf_path, page_index, dpi, out_path)"""
    pdf_path, page_index, dpi, out_path = task
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_index)
        pix = page.get_pixmap(dpi=dpi)
        pix.save(out_path)
        doc.close()
        return (page_index, out_path, None)
    except Exception as e:
        return (page_index, None, str(e))


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

    # >>> ALTERAÇÃO: carrega bolhas da imagem específica (fallback p/ global e padrão)
    bolhas_esq, bolhas_dir = carregar_bolhas_da_imagem(caminho_imagem, usar_fallback_global=True)

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
    caminho_saida = os.path.join(pASTA_RESULT if (pasta_saida is None) else pasta_saida, f"{base}_corrigido.png")
    # Corrige variável (caso pasta_saida seja None, mantém compatibilidade)
    if pasta_saida is None:
        pasta_saida = PASTA_RESULT
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


def converter_pdf_para_imagens(pdf_path, progress_cb=None, workers=None, dpi=300, cancelled=None):
    """
    Converte páginas do PDF em imagens em paralelo usando ProcessPoolExecutor.
    - pdf_path: caminho do PDF
    - progress_cb(atual, total): callback para atualizar UI
    - workers: número de processos (None -> os.cpu_count()-1)
    - cancelled: dicionário {"flag": False} que, se True, interrompe o processo.
    """
    os.makedirs(PASTA_TEMP, exist_ok=True)

    # Abre para descobrir número de páginas
    try:
        doc = fitz.open(pdf_path)
        total = len(doc)
        doc.close()
    except Exception as e:
        raise RuntimeError(f"Erro ao abrir PDF: {e}")

    if total == 0:
        return True

    if workers is None:
        cpu = os.cpu_count() or 1
        # normalmente deixamos 1 núcleo livre para o sistema/GUI
        workers = max(1, cpu - 1)

    tasks = []
    for i in range(total):
        img_path = os.path.join(PASTA_TEMP, f"pagina_{i+1}.png")
        tasks.append((pdf_path, i, dpi, img_path))

    # Execução paralela
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_index = {executor.submit(_render_page_to_png, t): t[1] for t in tasks}
            completed = 0
            for fut in concurrent.futures.as_completed(future_to_index):
                if cancelled and cancelled.get("flag"):
                    # tenta cancelar futuros pendentes
                    try:
                        executor.shutdown(wait=False, cancel_futures=True)
                    except Exception:
                        pass
                    break
                res = fut.result()
                completed += 1
                # res = (page_index, out_path, error_str_or_None)
                page_index, out_path, err = res
                if err:
                    print(f"Erro na página {page_index+1}: {err}")
                if progress_cb:
                    progress_cb(completed, total)
        # >>> NOVO: inicializa arquivos de bolhas por página (templates)
        criados = inicializar_bolhas_para_todas_as_imagens()
        if DEBUG_LOG:
            print(f"Arquivos de bolhas iniciais criados: {criados}")
        return True
    except Exception as e:
        # fallback sequencial caso o paralelismo falhe
        print("Falha no paralelismo, revertendo para conversão sequencial:", e)
        try:
            doc = fitz.open(pdf_path)
            for i in range(len(doc)):
                if cancelled and cancelled.get("flag"):
                    break
                pagina = doc.load_page(i)
                pix = pagina.get_pixmap(dpi=dpi)
                img_path = os.path.join(PASTA_TEMP, f"pagina_{i+1}.png")
                pix.save(img_path)
                if progress_cb:
                    progress_cb(i+1, len(doc))
            doc.close()
            # >>> NOVO: inicializa arquivos de bolhas por página (templates)
            criados = inicializar_bolhas_para_todas_as_imagens()
            if DEBUG_LOG:
                print(f"Arquivos de bolhas iniciais criados: {criados}")
            return True
        except Exception as e2:
            raise RuntimeError(f"Conversão sequencial também falhou: {e2}")


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
# EDITOR VISUAL DE BOLHAS (por página)
# =============================
class BubbleEditorDialog(QDialog):
    """
    Editor visual para reposicionar bolhas, agora percorrendo TODAS as imagens da PASTA_TEMP.
    Para cada imagem salva um arquivo específico: <nome>_bolhas.txt
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reposicionar Bolhas (por página)")
        self.setMinimumSize(1000, 760)

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

        # Controles de movimento
        controls = QHBoxLayout()

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

        grp_actions = QGroupBox("Ações (página atual)")
        ac = QHBoxLayout()
        self.btn_reset = QPushButton("Resetar para padrão (esta página)")
        self.btn_reset.clicked.connect(self.reset_layout)
        self.btn_save = QPushButton("Salvar")
        self.btn_save.clicked.connect(self.salvar_layout)
        self.btn_save_next = QPushButton("Salvar e Próxima")
        self.btn_save_next.clicked.connect(self.salvar_e_proxima)
        ac.addWidget(self.btn_reset)
        ac.addWidget(self.btn_save)
        ac.addWidget(self.btn_save_next)
        grp_actions.setLayout(ac)

        controls.addWidget(grp_move, stretch=2)
        controls.addWidget(grp_actions, stretch=3)
        main_layout.addLayout(controls)

        info = QLabel("Dica: use as setas do teclado para mover. A visualização usa o recorte definido em CROP.\n"
                      "Cada página terá seu próprio arquivo *_bolhas.txt em imagens_pdf/temp/.")
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
            self.info_crop_invalido = True
            self.lbl_preview.setPixmap(QPixmap())
            QMessageBox.warning(self, "CROP fora da imagem",
                                "A imagem é menor que a área de recorte definida em CROP.\n"
                                "Use uma imagem gerada pela conversão do PDF ou ajuste CROP.")
            self._atualizar_botoes_navegacao()
            return

        self.info_crop_invalido = False
        area = img_resized[y0:y1, x0:x1]
        self.preview = area.copy()

        # Carrega bolhas da página (ou global/padrão)
        self.bolhas_esq, self.bolhas_dir = carregar_bolhas_da_imagem(self.img_path_atual, usar_fallback_global=True)

        # Status
        arq_pag = caminho_arquivo_bolhas_da_imagem(self.img_path_atual)
        status = f"[{self.idx+1}/{len(self.imagens)}] {os.path.basename(self.img_path_atual)} | "
        if os.path.exists(arq_pag):
            status += f"Usando arquivo específico: {os.path.basename(arq_pag)}"
        elif os.path.exists(ARQUIVO_BOLHAS):
            status += f"Sem arquivo específico -> usando global: {ARQUIVO_BOLHAS}"
        else:
            status += "Sem arquivo específico -> usando LAYOUT PADRÃO"
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
        if not self.img_path_atual:
            return
        arq = salvar_bolhas_da_imagem(self.bolhas_esq, self.bolhas_dir, self.img_path_atual)
        QMessageBox.information(self, "Salvo", f"Coordenadas desta página gravadas em:\n{arq}")

    def salvar_e_proxima(self):
        self.salvar_layout()
        self.proxima_imagem()

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
        self.setGeometry(200, 200, 620, 580)

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
            dlg.setLabelText(f"{titulo}\nProgresso: {atual}/{total}\n"
                             f"Tempo decorrido: {int(elapsed)}s | Estimado restante: {int(restante)}s")
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

        # Pergunta ao usuário quantos workers usar (opcional) - sugerimos cpu_count-1 por padrão
        cpu = os.cpu_count() or 1
        default_workers = max(1, cpu - 1)
        n, ok = QInputDialog.getInt(self, "Número de processos",
                                    f"Informe o número de processos (recomendado {default_workers}):",
                                    value=default_workers, min=1, max=max(1, cpu))
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

        self._run_with_progress("Convertendo PDF para imagens (TEMP)...", 1, worker)
        # Informação extra: arquivos de bolhas por página já foram criados como templates
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

# =============================
# MAIN
# =============================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
