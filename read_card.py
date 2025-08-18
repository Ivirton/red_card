import cv2
import numpy as np

# 1. Carrega e redimensiona a imagem

pagina1 = 'imagens_pdf/temp/pagina_1.png'


gabarito_correto = [
    'C', 'D', 'C', 'D', 'C','D', 'A', 'B', 'E', 'C',
    'B', 'C', 'D', 'C', 'D','E', 'C', 'C', 'B', 'A'
]
imagem = cv2.imread(pagina1)
if imagem is None:
    raise ValueError("Imagem não encontrada!")

altura_max, largura_max = 1000, 800
alt, larg = imagem.shape[:2]
escala = min(altura_max / alt, largura_max / larg, 1.0)
nova_imagem = cv2.resize(imagem, (int(larg * escala), int(alt * escala)))

altura, largura = nova_imagem.shape[:2]

# Define manualmente a área de crop central (somente a parte do gabarito de bolhas)
# Esses valores foram estimados visualmente e funcionam bem para esse layout
y_inicio = 350   # linha inicial (vertical)
y_fim = 700     # linha final
x_inicio = 140   # coluna inicial (horizontal)
x_fim = 540      # coluna final
imagem_original = nova_imagem
# Faz o recorte
nova_imagem = nova_imagem[y_inicio:y_fim, x_inicio:x_fim]

#lista da coordenadas bolhas na coluna esquerda 
v_col_esq = [79, 101, 125, 150, 174]
v_lin_esq = [21, 48, 74, 99, 126, 151, 176, 200, 227, 251]
tamBolha = 20

bolhas_esquerda = [
	(v_col_esq[0],v_lin_esq[0], tamBolha, tamBolha),(v_col_esq[1], v_lin_esq[0], tamBolha, tamBolha),(v_col_esq[2], v_lin_esq[0], tamBolha, tamBolha),(v_col_esq[3],v_lin_esq[0], tamBolha, tamBolha),(v_col_esq[4],v_lin_esq[0], tamBolha, tamBolha),
	(v_col_esq[0],v_lin_esq[1], tamBolha, tamBolha),(v_col_esq[1], v_lin_esq[1], tamBolha, tamBolha),(v_col_esq[2], v_lin_esq[1], tamBolha, tamBolha),(v_col_esq[3],v_lin_esq[1], tamBolha, tamBolha),(v_col_esq[4],v_lin_esq[1], tamBolha, tamBolha),
	(v_col_esq[0],v_lin_esq[2], tamBolha, tamBolha),(v_col_esq[1], v_lin_esq[2], tamBolha, tamBolha),(v_col_esq[2], v_lin_esq[2], tamBolha, tamBolha),(v_col_esq[3],v_lin_esq[2], tamBolha, tamBolha),(v_col_esq[4],v_lin_esq[2], tamBolha, tamBolha), 
	(v_col_esq[0],v_lin_esq[3], tamBolha, tamBolha),(v_col_esq[1], v_lin_esq[3], tamBolha, tamBolha),(v_col_esq[2], v_lin_esq[3], tamBolha, tamBolha),(v_col_esq[3],v_lin_esq[3], tamBolha, tamBolha),(v_col_esq[4],v_lin_esq[3], tamBolha, tamBolha),
	(v_col_esq[0],v_lin_esq[4], tamBolha, tamBolha),(v_col_esq[1], v_lin_esq[4], tamBolha, tamBolha),(v_col_esq[2], v_lin_esq[4], tamBolha, tamBolha),(v_col_esq[3],v_lin_esq[4], tamBolha, tamBolha),(v_col_esq[4],v_lin_esq[4], tamBolha, tamBolha), 
	(v_col_esq[0],v_lin_esq[5], tamBolha, tamBolha),(v_col_esq[1], v_lin_esq[5], tamBolha, tamBolha),(v_col_esq[2], v_lin_esq[5], tamBolha, tamBolha),(v_col_esq[3],v_lin_esq[5], tamBolha, tamBolha),(v_col_esq[4],v_lin_esq[5], tamBolha, tamBolha),
	(v_col_esq[0],v_lin_esq[6], tamBolha, tamBolha),(v_col_esq[1], v_lin_esq[6], tamBolha, tamBolha),(v_col_esq[2], v_lin_esq[6], tamBolha, tamBolha),(v_col_esq[3],v_lin_esq[6], tamBolha, tamBolha),(v_col_esq[4],v_lin_esq[6], tamBolha, tamBolha), 
	(v_col_esq[0],v_lin_esq[7], tamBolha, tamBolha),(v_col_esq[1], v_lin_esq[7], tamBolha, tamBolha),(v_col_esq[2], v_lin_esq[7], tamBolha, tamBolha),(v_col_esq[3],v_lin_esq[7], tamBolha, tamBolha),(v_col_esq[4],v_lin_esq[7], tamBolha, tamBolha), 
	(v_col_esq[0],v_lin_esq[8], tamBolha, tamBolha),(v_col_esq[1], v_lin_esq[8], tamBolha, tamBolha),(v_col_esq[2], v_lin_esq[8], tamBolha, tamBolha),(v_col_esq[3],v_lin_esq[8], tamBolha, tamBolha),(v_col_esq[4],v_lin_esq[8], tamBolha, tamBolha), 
	(v_col_esq[0],v_lin_esq[9], tamBolha, tamBolha),(v_col_esq[1], v_lin_esq[9], tamBolha, tamBolha),(v_col_esq[2], v_lin_esq[9], tamBolha, tamBolha),(v_col_esq[3],v_lin_esq[9], tamBolha, tamBolha),(v_col_esq[4],v_lin_esq[9], tamBolha, tamBolha)
]


# lista da coordenadas bolhas  da coluna da direita
v_col_dir = [262, 286, 311, 336, 360]
v_lin_dir = [22, 46, 72, 97, 124, 149, 174, 200, 227, 251]
bolhas_direita = [
    (262, 22, 20, 20),  (286, 22, 21, 20),  (311, 22, 20, 20),  (336, 22, 20, 20),  (360, 22, 20, 20),
    (262, 46, 20, 20),  (286, 46, 21, 20),  (311, 46, 21, 20),  (336, 46, 20, 20),  (360, 46, 20, 20),
    (262, 72, 20, 20),  (286, 72, 20, 20),  (311, 72, 20, 20),  (336, 72, 20, 20),  (360, 72, 20, 20),
    (262, 97, 20, 20),  (286, 97, 21, 20),  (311, 97, 20, 20),  (336, 97, 20, 20),  (360, 97, 20, 20),
    (262, 124, 20, 20), (286, 124, 20, 20), (311, 124, 20, 20), (336, 123, 20, 20), (360, 124, 20, 20), 
    (262, 149, 20, 20), (286, 149, 20, 20), (311, 149, 20, 20), (336, 149, 21, 20), (360, 149, 20, 20),
    (262, 174, 20, 20), (286, 174, 21, 20), (311, 174, 20, 20), (336, 174, 20, 20), (360, 174, 20, 20),
    (262, 200, 20, 20), (286, 200, 20, 20), (311, 200, 20, 20), (336, 200, 21, 20), (360, 200, 20, 20),
    (262, 227, 20, 20), (286, 227, 20, 20), (311, 227, 20, 20), (336, 227, 20, 20), (360, 227, 20, 20),
    (262, 251, 20, 20), (286, 252, 20, 20), (311, 252, 20, 20), (336, 252, 20, 20), (360, 252, 20, 20),
]


# 2. Pré-processamento
cinza = cv2.cvtColor(nova_imagem, cv2.COLOR_BGR2GRAY)
borrada = cv2.GaussianBlur(cinza, (5, 5), 0)
_, thresh = cv2.threshold(borrada, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 3. Encontra contornos
contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 4. Filtra contornos parecidos com bolhas (círculos)
bolhas = []
for cnt in contornos:
    (x, y, w, h) = cv2.boundingRect(cnt)
    aspecto = w / float(h)
    if 1 < w < 60 and 15 < h < 60 and 0.8 <= aspecto <= 1.2:
        bolhas.append((x, y, w, h))

# 5. Organiza bolhas em colunas
coluna_esquerda = []
coluna_direita = []
for b in bolhas:
    if b[0] < nova_imagem.shape[1] // 2:
        coluna_esquerda.append(b)
    else:
        coluna_direita.append(b)

# 6. Desenha retângulos nas bolhas detectadas
imagem_saida = nova_imagem.copy()
for x, y, w, h in  bolhas_esquerda + bolhas_direita :
    cv2.rectangle(imagem_saida, (x, y), (x + w, y + h), (255, 0, 0), 1)

for x, y, w, h in  coluna_esquerda + coluna_direita:
    cv2.rectangle(imagem_saida, (x, y), (x + w, y + h), (0, 255, 0), 2)

coluna_esquerda = coluna_esquerda[::-1]  # Inverte a ordem para exibição correta
coluna_direita = coluna_direita[::-1]  # Inverte a ordem para exibição correta


# Função para verificar se uma bolha do gabarito foi marcada (tem contorno próximo)
def bolha_foi_marcada(bolha_gabarito, bolhas_detectadas, margem=10):
    gx, gy, gw, gh = bolha_gabarito
    for bx, by, bw, bh in bolhas_detectadas:
        if abs(gx - bx) < margem and abs(gy - by) < margem:
            return True
    return False

def identificar_marcadas(bolhas_gabarito, bolhas_detectadas, lado="esquerdo"):
    respostas = []
    letras = ['A', 'B', 'C', 'D', 'E']
    
    for i in range(0, len(bolhas_gabarito), 5):
        grupo = bolhas_gabarito[i:i+5]
        marcadas = []

        for j, bolha in enumerate(grupo):
            if bolha_foi_marcada(bolha, bolhas_detectadas):
                marcadas.append(letras[j])

        if len(marcadas) == 0:
            respostas.append("-")  # Nenhuma marcada
        elif len(marcadas) == 1:
            respostas.append(marcadas[0])  # Uma marcada
        else:
            respostas.append("*")  # Múltiplas marcadas → ANULADA

    return respostas

respostas_esquerda = identificar_marcadas(bolhas_esquerda, coluna_esquerda, lado="esquerdo")
respostas_direita = identificar_marcadas(bolhas_direita, coluna_direita, lado="direito")

# Combina as respostas
respostas_final = respostas_esquerda + respostas_direita

# === COMPARA COM GABARITO E MOSTRA RESULTADO ===
print("\n📋 RESULTADO FINAL:")
acertos = 0
anuladas = 0
brancas = 0

for i, resposta_aluno in enumerate(respostas_final):
    resposta_correta = gabarito_correto[i]

    if resposta_aluno == "*":
        status = "❌ Questão ANULADA (múltiplas)"
        anuladas += 1
    elif resposta_aluno == "-":
        status = "⚪ Questão EM BRANCO"
        brancas += 1
    elif resposta_aluno == resposta_correta:
        status = "✔️ CORRETA"
        acertos += 1
    else:
        status = f"❌ ERRADA (correta: {resposta_correta})"

    print(f"Questão {i+1:02}: Marcou [{resposta_aluno}] → {status}")

print(f"\n✅ Total de acertos: {acertos}/20")
print(f"❌ Anuladas (múltiplas): {anuladas}")
print(f"⚪ Em branco: {brancas}")


# Cria uma cópia da imagem original para sobreposição
imagem_recombinada = imagem_original.copy()

# Substitui a área recortada pela imagem com marcações (imagem_saida)
imagem_recombinada[y_inicio:y_fim, x_inicio:x_fim] = imagem_saida

cv2.imshow('Imagem Recombinada', imagem_recombinada)

# cv2.imshow('imagem', imagem_original)
# cv2.imshow('Bolhas Detectadas', imagem_saida)
cv2.waitKey(0)
cv2.destroyAllWindows()
