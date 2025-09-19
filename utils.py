import time
import os
import numpy as np
from pathlib import Path
import shutil
import sys
import cv2
import fitz
import concurrent.futures

from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QProgressDialog, QApplication, QMessageBox
from PyQt6.QtCore import Qt

from config.config import *

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
    return GABARITO_CORRETO.copy()

def default_bolhas_layout():
    """Gera layout padrão (50 esq + 50 dir). Retorna (esq_list, dir_list)."""
    v_col_esq = [79, 101, 125, 150, 174]
    v_lin_esq = [21, 48, 74, 99, 126, 151, 176, 200, 227, 251]
    tam = 20
    bolhas_esq = [(x, y, tam, tam) for y in v_lin_esq for x in v_col_esq]

    v_col_dir = [262, 286, 311, 336, 360]
    v_lin_dir = [22, 46, 72, 97, 124, 149, 174, 200, 227, 251]
    larguras_dir = {(286, 22): 21, (311, 46): 21, (336, 123): 20, (336, 149): 21, (286, 174): 21, (336, 200): 21}
    bolhas_dir = [(x, y, larguras_dir.get((x, y), 20), 20) for y in v_lin_dir for x in v_col_dir]
    return bolhas_esq, bolhas_dir


def salvar_bolhas(bolhas_esq, bolhas_dir, arquivo=ARQUIVO_BOLHAS, lock=False):
    """Salva EM ARQUIVO informado (global ou por imagem). Se lock=True, escreve cabeçalho #LOCK=1."""
    os.makedirs(os.path.dirname(arquivo) or ".", exist_ok=True)
    with open(arquivo, "w", encoding="utf-8") as f:
        if lock:
            f.write("#LOCK=1\n")
        for (x, y, w, h) in bolhas_esq:
            f.write(f"E,{x},{y},{w},{h}\n")
        for (x, y, w, h) in bolhas_dir:
            f.write(f"D,{x},{y},{w},{h}\n")


def carregar_bolhas(arquivo=ARQUIVO_BOLHAS):
    """Lê arquivo de bolhas no formato E,x,y,w,h / D,x,y,w,h, ignora linhas iniciadas por #."""
    if not os.path.exists(arquivo):
        return default_bolhas_layout()
    esq, dir_ = [], []
    with open(arquivo, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
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


def arquivo_tem_lock(arquivo):
    """Retorna True se o arquivo possui cabeçalho #LOCK=1."""
    if not os.path.exists(arquivo):
        return False
    try:
        with open(arquivo, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i > 3:
                    break
                if line.strip().upper() == "#LOCK=1":
                    return True
    except Exception:
        pass
    return False


def caminho_arquivo_bolhas_da_imagem(img_path):
    """Retorna o caminho do arquivo de bolhas específico para a imagem/página."""
    base = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(PASTA_TEMP, f"{base}_bolhas.txt")


def carregar_bolhas_da_imagem(img_path, usar_fallback_global=True):
    """Carrega bolhas específicas da imagem; fallback para global e padrão. (ignora #LOCK na leitura)"""
    arq_pagina = caminho_arquivo_bolhas_da_imagem(img_path)
    if os.path.exists(arq_pagina):
        return carregar_bolhas(arq_pagina)
    if usar_fallback_global and os.path.exists(ARQUIVO_BOLHAS):
        return carregar_bolhas(ARQUIVO_BOLHAS)
    return default_bolhas_layout()


def pagina_bloqueada_por_usuario(img_path):
    """True se o _bolhas.txt da página tiver #LOCK=1 (salvo pelo editor)."""
    arq_pagina = caminho_arquivo_bolhas_da_imagem(img_path)
    return arquivo_tem_lock(arq_pagina)


def salvar_bolhas_da_imagem(bolhas_esq, bolhas_dir, img_path, lock=True):
    """Salva as bolhas específicas daquela imagem/página. lock=True para marcar que o usuário travou."""
    arq_pagina = caminho_arquivo_bolhas_da_imagem(img_path)
    salvar_bolhas(bolhas_esq, bolhas_dir, arq_pagina, lock=lock)
    return arq_pagina


def inicializar_bolhas_para_todas_as_imagens():
    """
    Cria (se não existirem) arquivos _bolhas.txt individuais, com layout padrão,
    para cada imagem presente em PASTA_TEMP (sem LOCK por padrão).
    """
    if not os.path.exists(PASTA_TEMP):
        return 0
    arquivos = sorted(
        [a for a in os.listdir(PASTA_TEMP) if a.lower().endswith((".png", ".jpg", ".jpeg"))]
    )
    bolhas_esq_default, bolhas_dir_default = default_bolhas_layout()
    criados = 0
    for nome in arquivos:
        img_path = os.path.join(PASTA_TEMP, nome)
        arq_pag = caminho_arquivo_bolhas_da_imagem(img_path)
        if not os.path.exists(arq_pag):
            salvar_bolhas(bolhas_esq_default, bolhas_dir_default, arq_pag, lock=False)
            criados += 1
    return criados


def aplicar_layout_a_todas_as_imagens(bolhas_esq, bolhas_dir):
    """
    Sobrescreve o *_bolhas.txt de TODAS as páginas com o layout fornecido
    e também salva como padrão global (ARQUIVO_BOLHAS). (aplica SEM lock)
    """
    salvar_bolhas(bolhas_esq, bolhas_dir, ARQUIVO_BOLHAS, lock=False)
    if not os.path.exists(PASTA_TEMP):
        return 0
    arquivos = sorted(
        [a for a in os.listdir(PASTA_TEMP) if a.lower().endswith((".png", ".jpg", ".jpeg"))]
    )
    alterados = 0
    for nome in arquivos:
        img_path = os.path.join(PASTA_TEMP, nome)
        arq_pag = caminho_arquivo_bolhas_da_imagem(img_path)
        salvar_bolhas(bolhas_esq, bolhas_dir, arq_pag, lock=False)
        alterados += 1
    return alterados


def cvimg_to_qpixmap(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape
    qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


# =============================
# DETECÇÃO / ALINHAMENTO
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
        return 0.0, 255.0

    blur = cv2.GaussianBlur(roi, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones(MORPH_KERNEL, np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    rh, rw = th.shape[:2]
    mask = np.zeros((rh, rw), dtype=np.uint8)
    center = (rw // 2, rh // 2)
    radius = max(1, min(rw, rh) // 2 - 1)
    cv2.circle(mask, center, radius, 255, -1)

    masked = cv2.bitwise_and(th, th, mask=mask)
    filled = cv2.countNonZero(masked)
    total_mask = cv2.countNonZero(mask)
    fill_ratio = (filled / total_mask) if total_mask > 0 else 0.0
    mean_gray = float(cv2.mean(roi, mask=mask)[0]) if total_mask > 0 else 255.0
    return float(fill_ratio), mean_gray


def _render_mask_from_bolhas(size_hw, bolhas_esq, bolhas_dir):
    """Gera uma máscara sintética (float32) com círculos nas posições das bolhas para alinhamento."""
    H, W = size_hw
    mask = np.zeros((H, W), dtype=np.uint8)

    def draw_group(lst):
        for (x, y, w, h) in lst:
            cx, cy = int(x + w / 2), int(y + h / 2)
            r = max(3, int(min(w, h) * 0.45))
            cv2.circle(mask, (cx, cy), r, 255, -1)

    draw_group(bolhas_esq)
    draw_group(bolhas_dir)
    # Desfoca um pouco para facilitar ECC
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask.astype(np.float32) / 255.0


def _preprocess_for_ecc(gray_area):
    """Pré-processa a área real para ECC: binariza e normaliza como float32 0..1."""
    blur = cv2.GaussianBlur(gray_area, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.GaussianBlur(th, (5, 5), 0)
    return th.astype(np.float32) / 255.0


def _transform_points(bolhas, M, motion=cv2.MOTION_AFFINE):
    """Aplica transformação (warp) nos centros e mantém tamanho original."""
    out = []
    if motion in (cv2.MOTION_AFFINE,):
        A = M[:, :2]
        b = M[:, 2].reshape(2, 1)
    else:
        # Euclidean: 2x3 também funciona como caso geral
        A = M[:, :2]
        b = M[:, 2].reshape(2, 1)
    for (x, y, w, h) in bolhas:
        p = np.array([[x + w / 2.0], [y + h / 2.0]], dtype=np.float32)
        p2 = A @ p + b
        cx, cy = float(p2[0, 0]), float(p2[1, 0])
        out.append((int(round(cx - w / 2.0)), int(round(cy - h / 2.0)), w, h))
    return out


def auto_reposicionar_bolhas(gray_area, bolhas_esq, bolhas_dir):
    """
    Realinha automaticamente as bolhas à imagem usando ECC (MOTION_AFFINE).
    Retorna: (bolhas_esq_alinhadas, bolhas_dir_alinhadas, sucesso_bool)
    """
    try:
        H, W = gray_area.shape[:2]
        ref = _render_mask_from_bolhas((H, W), bolhas_esq, bolhas_dir)
        tgt = _preprocess_for_ecc(gray_area)

        # ECC precisa float32, mesmo tamanho
        ref32 = ref
        tgt32 = tgt

        # Matriz inicial (identidade)
        warp = np.eye(2, 3, dtype=np.float32)

        # Critérios e iterações
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5)

        # Tenta AFFINE (permite rotação + leve escala)
        cc, warp = cv2.findTransformECC(
            ref32, tgt32, warp, motionType=cv2.MOTION_AFFINE,
            criteria=criteria, inputMask=None, gaussFiltSize=5
        )

        # Aplica transformação às bolhas
        esq_al = _transform_points(bolhas_esq, warp, motion=cv2.MOTION_AFFINE)
        dir_al = _transform_points(bolhas_dir, warp, motion=cv2.MOTION_AFFINE)
        return esq_al, dir_al, True
    except Exception as e:
        if DEBUG_LOG:
            print("Auto-reposicionamento falhou:", e)
        return bolhas_esq, bolhas_dir, False


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

    # Redimensiona
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

    # Carrega bolhas base (padrão/global/por-página)
    bolhas_esq_base, bolhas_dir_base = carregar_bolhas_da_imagem(caminho_imagem, usar_fallback_global=True)

    # Se o usuário editou e SALVOU essa página (arquivo com #LOCK=1), NÃO auto-reposiciona.
    usar_auto = not pagina_bloqueada_por_usuario(caminho_imagem)

    # Auto-reposicionamento (alinha com AFFINE via ECC)
    if usar_auto:
        bolhas_esq, bolhas_dir, ok_auto = auto_reposicionar_bolhas(gray_area, bolhas_esq_base, bolhas_dir_base)
    else:
        bolhas_esq, bolhas_dir, ok_auto = bolhas_esq_base, bolhas_dir_base, False

    # Calcula fill ratio
    fill_esq = [calc_fill_ratio(gray_area, x, y, w, h) for (x, y, w, h) in bolhas_esq]
    fill_dir = [calc_fill_ratio(gray_area, x, y, w, h) for (x, y, w, h) in bolhas_dir]

    letras = ['A', 'B', 'C', 'D', 'E']
    respostas_esq, respostas_dir = [], []
    logs = []

    # esquerda (1..10)
    for g in range(0, 50, 5):
        group = fill_esq[g:g+5]
        vals = [v[0] for v in group]
        pairs = list(zip(letras, vals, [v[1] for v in group]))
        sorted_pairs = sorted(enumerate(vals), key=lambda t: t[1], reverse=True)
        top_idx = sorted_pairs[0][0]
        top_fill = vals[top_idx]
        second_fill = sorted_pairs[1][1] if len(sorted_pairs) > 1 else 0.0

        if top_fill < FILL_MIN:
            decision = "-"
            reason = f"top_fill {top_fill:.3f} < FILL_MIN {FILL_MIN}"
        elif (top_fill - second_fill) < DIFF_MIN:
            decision = "*"
            reason = f"dif {top_fill - second_fill:.3f} < DIFF_MIN {DIFF_MIN}"
        else:
            decision = letras[top_idx]
            reason = f"top {top_fill:.3f} ok (2º {second_fill:.3f})"
        respostas_esq.append(decision)
        logs.append(("esq", (g//5)+1, pairs, decision, reason))

    # direita (11..20)
    for g in range(0, 50, 5):
        group = fill_dir[g:g+5]
        vals = [v[0] for v in group]
        pairs = list(zip(letras, vals, [v[1] for v in group]))
        sorted_pairs = sorted(enumerate(vals), key=lambda t: t[1], reverse=True)
        top_idx = sorted_pairs[0][0]
        top_fill = vals[top_idx]
        second_fill = sorted_pairs[1][1] if len(sorted_pairs) > 1 else 0.0

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
        logs.append(("dir", (g//5)+11, pairs, decision, reason))

    respostas_final = respostas_esq + respostas_dir  # 20 respostas

    # compara com gabarito e desenha
    acertos = anuladas = brancas = 0
    imagem_saida = area.copy()

    # [NOVO] Desenha a grade completa (todas as bolhas), para facilitar validação visual
    if DESENHAR_GRADE_TODAS_BOLHAS:
        for (x, y, w, h) in bolhas_esq + bolhas_dir:
            cv2.rectangle(imagem_saida, (x, y), (x + w, y + h), COR_GRADE, ESPESSURA_GRADE)

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

        # grupo e alternativa
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

    # recombina imagem final
    imagem_final = imagem_original.copy()
    imagem_final[CROP["y_inicio"]:CROP["y_fim"], CROP["x_inicio"]:CROP["x_fim"]] = imagem_saida

    # escreve resumo (e status do auto)
    resumo = [
        f"Acertos: {acertos}/{len(gabarito)}",
        f"Anuladas: {anuladas}",
        f"Em branco: {brancas}",
        f"Auto-reposicionamento: {'OK' if usar_auto and ok_auto else ('IGNORADO (LOCK)' if not usar_auto else 'FALHOU')}"
    ]
    x_texto, y_texto = 10, imagem_final.shape[0] - 100
    for idx, linha in enumerate(resumo):
        cv2.putText(imagem_final, linha, (x_texto, y_texto + idx*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # salva arquivo final e log
    os.makedirs(pasta_saida, exist_ok=True)
    base, _ = os.path.splitext(os.path.basename(caminho_imagem))
    caminho_saida = os.path.join(pasta_saida, f"{base}_corrigido.png")
    cv2.imwrite(caminho_saida, imagem_final)

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
# PROCESSAMENTO EM LOTE E PDF
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
    """
    os.makedirs(PASTA_TEMP, exist_ok=True)

    # Descobre número de páginas
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
        workers = max(1, cpu - 1)

    tasks = []
    for i in range(total):
        img_path = os.path.join(PASTA_TEMP, f"pagina_{i+1}.png")
        tasks.append((pdf_path, i, dpi, img_path))

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_index = {executor.submit(_render_page_to_png, t): t[1] for t in tasks}
            completed = 0
            for fut in concurrent.futures.as_completed(future_to_index):
                if cancelled and cancelled.get("flag"):
                    try:
                        executor.shutdown(wait=False, cancel_futures=True)
                    except Exception:
                        pass
                    break
                res = fut.result()
                completed += 1
                page_index, out_path, err = res
                if err:
                    print(f"Erro na página {page_index+1}: {err}")
                if progress_cb:
                    progress_cb(completed, total)
        criados = inicializar_bolhas_para_todas_as_imagens()
        if DEBUG_LOG:
            print(f"Arquivos de bolhas iniciais criados: {criados}")
        return True
    except Exception as e:
        # fallback sequencial
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
# OUTROS UTILITÁRIOS
# =============================

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