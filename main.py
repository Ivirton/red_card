import cv2
import numpy as np

# 1. Carrega e redimensiona a imagem
todasbolhas = 'todasbolhas.png'
pagina1 = 'imagens_pdf/pagina_1.png'
pagina2 = 'imagens_pdf/pagina_2.png'
gabarito_correto = [
    'C', 'D', 'C', 'D', 'C','D', 'A', 'B', 'E', 'C',
    'B', 'C', 'D', 'C', 'D','E', 'C', 'C', 'B', 'A'
]
imagem = cv2.imread(pagina2)
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

bolhas_esquerda = [
	(79, 21, 20, 20),  (101, 21, 20, 20),  (125, 21, 20, 20),  (150, 21, 20, 20),  (174, 21, 20, 20),
	(79, 48, 20, 20),  (101, 48, 20, 20),  (125, 48, 20, 20),  (150, 48, 20, 20),  (174, 48, 20, 20),
	(79, 74, 20, 20),  (101, 74, 20, 20),  (125, 74, 20, 20),  (150, 74, 20, 20),  (174, 74, 20, 20), 
	(79, 99, 20, 20),  (101, 99, 20, 20),  (125, 99, 20, 20),  (150, 99, 20, 20),  (174, 99, 20, 20),
	(79, 126, 20, 20), (101, 126, 20, 20), (125, 126, 20, 20), (150, 125, 20, 20), (174, 126, 20, 20), 
	(79, 151, 20, 20), (101, 151, 20, 20), (125, 151, 20, 20), (150, 151, 20, 20), (174, 151, 20, 20),
	(79, 176, 20, 20), (101, 176, 20, 20), (125, 176, 20, 20), (150, 176, 20, 20), (174, 176, 20, 20), 
	(79, 200, 20, 20), (101, 200, 20, 20), (125, 200, 20, 20), (150, 200, 20, 20), (174, 200, 20, 20), 
	(79, 227, 20, 20), (101, 227, 20, 20), (125, 227, 20, 20), (150, 227, 20, 20), (174, 227, 20, 20), 
	(79, 251, 20, 20), (101, 251, 20, 20), (125, 251, 20, 20), (150, 251, 20, 20), (174, 251, 20, 20)
]


# lista da coordenadas bolhas  da coluna da direita

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
        resposta = "-"
        for j, bolha in enumerate(grupo):
            if bolha_foi_marcada(bolha, bolhas_detectadas):
                resposta = letras[j]
                break  # assume apenas uma marca por linha
        respostas.append(resposta)
    return respostas
respostas_esquerda = identificar_marcadas(bolhas_esquerda, coluna_esquerda, lado="esquerdo")
respostas_direita = identificar_marcadas(bolhas_direita, coluna_direita, lado="direito")

# Combina as respostas
respostas_final = respostas_esquerda + respostas_direita

# === COMPARA COM GABARITO E MOSTRA RESULTADO ===
print("\n📋 RESULTADO FINAL:")
acertos = 0

for i, resposta_aluno in enumerate(respostas_final):
    resposta_correta = gabarito_correto[i]
    simbolo = "✔️" if resposta_aluno == resposta_correta else "❌"
    if resposta_aluno == resposta_correta:
        acertos += 1
    print(f"Questão {i+1:02}: Marcou [{resposta_aluno}] - Correta: [{resposta_correta}] {simbolo}")

print(f"\n✅ Total de acertos: {acertos}/20")
# print(f"📝 Nota final: {acertos * 5} pontos")  # ou multiplique por 6, se preferir

cv2.imshow('imagem', imagem_original)
cv2.imshow('Bolhas Detectadas', imagem_saida)
cv2.waitKey(0)
cv2.destroyAllWindows()
