import fitz  # PyMuPDF
import os

# Caminho para o PDF
pdf_path = "gabarito2.pdf"

# Pasta para salvar as imagens
output_folder = "imagens_pdf"
os.makedirs(output_folder, exist_ok=True)

# Abre o PDF
pdf = fitz.open(pdf_path)

# Itera sobre cada página
for i in range(len(pdf)):
    pagina = pdf.load_page(i)  # Carrega a página
    pix = pagina.get_pixmap(dpi=300)  # Converte a página em imagem
    imagem_path = os.path.join(output_folder, f"pagina_{i + 1}.png")
    pix.save(imagem_path)
    print(f"Salvo: {imagem_path}")

pdf.close()
