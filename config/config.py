# ======================================
# CONFIGURAÇÕES GLOBAIS / PARÂMETROS
# ======================================

PASTA_TEMP = "imagens_pdf/temp"
PASTA_RESULT = "imagens_pdf/result"
ARQUIVO_GABARITO = "gabarito.txt"
ARQUIVO_BOLHAS = "bolhas.txt"  # layout global

# Área de recorte da região das respostas (ajuste se necessário)
CROP = dict(y_inicio=350, y_fim=700, x_inicio=140, x_fim=540)

# Gabarito padrão (20 questões)
GABARITO_CORRETO = [
    'A', 'C', 'C', 'D', 'C', 'D', 'A', 'B', 'E', 'C',
    'B', 'C', 'D', 'C', 'D', 'E', 'C', 'C', 'B', 'A'
]

# Parâmetros de decisão
FILL_MIN = 0.18      # mínimo para considerar bolha preenchida
DIFF_MIN = 0.12      # diferença entre top e 2º para aceitar top
ROI_EXPAND = 2       # px ao redor do ROI
MORPH_KERNEL = (3, 3)
DEBUG_LOG = False

# Overlays
DESENHAR_GRADE_TODAS_BOLHAS = True         # desenha todos os “quadradinhos/grades” por cima
COR_GRADE = (0, 200, 255)                   # cor da grade das alternativas (BGR)
ESPESSURA_GRADE = 1                         # espessura linha grade