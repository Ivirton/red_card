import cv2
import numpy as np
import os
import fitz  # PyMuPDF

# === CONFIGURAÇÕES ===
pdf_path = "gabarito2.pdf"              # PDF com os cartões de resposta
pasta_temp = "imagens_pdf/temp"         # páginas convertidas do PDF
pasta_result = "imagens_pdf/result"     # imagens processadas
saida_pdf = "imagens_pdf/cartoes_corrigidos.pdf"

os.makedirs(pasta_temp, exist_ok=True)
os.makedirs(pasta_result, exist_ok=True)

# === 1. Converter PDF em imagens (uma página = uma imagem) ===
print("📄 Convertendo PDF em imagens...")
pdf = fitz.open(pdf_path)
for i in range(len(pdf)):
    pagina = pdf.load_page(i)
    pix = pagina.get_pixmap(dpi=300)
    imagem_path = os.path.join(pasta_temp, f"pagina_{i + 1}.png")
    pix.save(imagem_path)
    print(f"   ✔️ Salvo: {imagem_path}")
pdf.close()

# === GABARITO (ajuste conforme sua prova) ===
gabarito_correto = [
    'A', 'C', 'C', 'D', 'C','D', 'A', 'B', 'E', 'C',
    'B', 'C', 'D', 'C', 'D','E', 'C', 'C', 'B', 'A'
]

# === FUNÇÕES AUXILIARES ===
def bolha_foi_marcada(bolha_gabarito, bolhas_detectadas, margem=10):
    gx, gy, gw, gh = bolha_gabarito
    for bx, by, bw, bh in bolhas_detectadas:
        if abs(gx - bx) < margem and abs(gy - by) < margem:
            return True
    return False

def identificar_marcadas(bolhas_gabarito, bolhas_detectadas):
    respostas = []
    letras = ['A', 'B', 'C', 'D', 'E']
    for i in range(0, len(bolhas_gabarito), 5):
        grupo = bolhas_gabarito[i:i+5]
        marcadas = []
        for j, bolha in enumerate(grupo):
            if bolha_foi_marcada(bolha, bolhas_detectadas):
                marcadas.append(letras[j])
        if len(marcadas) == 0:
            respostas.append("-")
        elif len(marcadas) == 1:
            respostas.append(marcadas[0])
        else:
            respostas.append("*")
    return respostas

def processar_cartao(caminho_imagem, gabarito, pasta_saida):
    """Processa UM cartão de respostas e salva a imagem corrigida"""
    imagem = cv2.imread(caminho_imagem)
    if imagem is None:
        print(f"⚠️ Erro: não consegui abrir {caminho_imagem}")
        return None
    
    # redimensiona
    altura_max, largura_max = 1000, 800
    alt, larg = imagem.shape[:2]
    escala = min(altura_max / alt, largura_max / larg, 1.0)
    nova_imagem = cv2.resize(imagem, (int(larg * escala), int(alt * escala)))

    # recorte da área de respostas
    y_inicio, y_fim = 350, 700
    x_inicio, x_fim = 140, 540
    imagem_original = nova_imagem.copy()
    area_respostas = nova_imagem[y_inicio:y_fim, x_inicio:x_fim]

    # coordenadas gabarito
    v_col_esq = [79, 101, 125, 150, 174]
    v_lin_esq = [21, 48, 74, 99, 126, 151, 176, 200, 227, 251]
    tamBolha = 20
    bolhas_esquerda = [(x, y, tamBolha, tamBolha) for y in v_lin_esq for x in v_col_esq]

    v_col_dir = [262, 286, 311, 336, 360]
    v_lin_dir = [22, 46, 72, 97, 124, 149, 174, 200, 227, 251]
    larguras_dir = {
        (286, 22): 21, (311, 46): 21, (336, 123): 20, (336, 149): 21,
        (286, 174): 21, (336, 200): 21
    }
    bolhas_direita = [
        (x, y, larguras_dir.get((x, y), 20), 20)
        for y in v_lin_dir
        for x in v_col_dir
    ]

    # pré-processamento
    cinza = cv2.cvtColor(area_respostas, cv2.COLOR_BGR2GRAY)
    borrada = cv2.GaussianBlur(cinza, (5, 5), 0)
    _, thresh = cv2.threshold(borrada, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bolhas = []
    for cnt in contornos:
        (x, y, w, h) = cv2.boundingRect(cnt)
        aspecto = w / float(h)
        if 1 < w < 60 and 15 < h < 60 and 0.8 <= aspecto <= 1.2:
            bolhas.append((x, y, w, h))

    coluna_esquerda = [b for b in bolhas if b[0] < area_respostas.shape[1] // 2][::-1]
    coluna_direita = [b for b in bolhas if b[0] >= area_respostas.shape[1] // 2][::-1]

    respostas_esquerda = identificar_marcadas(bolhas_esquerda, coluna_esquerda)
    respostas_direita = identificar_marcadas(bolhas_direita, coluna_direita)
    respostas_final = respostas_esquerda + respostas_direita

    # comparação com gabarito
    acertos, anuladas, brancas = 0, 0, 0
    letras = ['A', 'B', 'C', 'D', 'E']
    imagem_saida = area_respostas.copy()

    for i, (resposta_aluno, resposta_correta) in enumerate(zip(respostas_final, gabarito)):
        if i < len(bolhas_esquerda) // 5:
            grupo = bolhas_esquerda[i*5:(i+1)*5]
        else:
            j = i - len(bolhas_esquerda)//5
            grupo = bolhas_direita[j*5:(j+1)*5]

        if resposta_aluno == "*":
            anuladas += 1
            continue
        elif resposta_aluno == "-":
            brancas += 1
            continue
        elif resposta_aluno == resposta_correta:
            acertos += 1
            cor = (0, 255, 0)  # verde
        else:
            cor = (0, 0, 255)  # vermelho

        idx_marcada = letras.index(resposta_aluno)
        x, y, w, h = grupo[idx_marcada]
        cv2.rectangle(imagem_saida, (x, y), (x + w, y + h), cor, 2)

    # recombina com imagem original
    imagem_final = imagem_original.copy()
    imagem_final[y_inicio:y_fim, x_inicio:x_fim] = imagem_saida

    # escreve resumo
    resumo = [
        f"Acertos: {acertos}/{len(gabarito)}",
        f"Anuladas: {anuladas}",
        f"Em branco: {brancas}"
    ]
    x_texto, y_texto = 10, imagem_final.shape[0] - 80
    for i, linha in enumerate(resumo):
        cv2.putText(imagem_final, linha,
                    (x_texto, y_texto + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # salva na pasta de saída
    nome_saida = os.path.basename(caminho_imagem).replace(".png", "_corrigido.png")
    caminho_saida = os.path.join(pasta_saida, nome_saida)
    cv2.imwrite(caminho_saida, imagem_final)

    print(f"   ✅ Processado: {caminho_imagem} -> {caminho_saida}")
    return caminho_saida

# === 2. Processar todas as imagens geradas do PDF ===
print("\n🖼️ Processando cartões de resposta...")
imagens_corrigidas = []
for arquivo in sorted(os.listdir(pasta_temp)):
    if arquivo.lower().endswith((".png", ".jpg", ".jpeg")):
        caminho_img = os.path.join(pasta_temp, arquivo)
        saida = processar_cartao(caminho_img, gabarito_correto, pasta_result)
        if saida:
            imagens_corrigidas.append(saida)

# === 3. Gerar PDF com todas as imagens corrigidas ===
print("\n📑 Gerando PDF final...")
doc = fitz.open()
for img_path in imagens_corrigidas:
    img = fitz.Pixmap(img_path)
    rect = fitz.Rect(0, 0, img.width, img.height)
    pagina = doc.new_page(width=img.width, height=img.height)
    pagina.insert_image(rect, pixmap=img)
doc.save(saida_pdf)
doc.close()

print("\n🏁 FIM! Todas as imagens foram processadas e salvas em:")
print(f"   📂 Pasta: {pasta_result}")
print(f"   📄 PDF final: {saida_pdf}")
