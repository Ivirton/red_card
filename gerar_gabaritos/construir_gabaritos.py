import cv2
import numpy as np
import qrcode 
from qrcode.image.pil import PilImage
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import sqlite3

from utils import run_with_progress

def creat_qr_code(data):
    qr = qrcode.QRCode(
        version=1, 
        error_correction=qrcode.constants.ERROR_CORRECT_L, 
        box_size=10, 
        border=4
    )

    qr.add_data(data)
    return qr.make_image(PilImage, fill_color="black", back_color="white")

def apply_qr_code(gabarito, dados):
    qr = creat_qr_code(dados) 
    qr = qr.convert("RGB")
    qr = np.array(qr)
    qr = cv2.cvtColor(qr, cv2.COLOR_RGB2BGR)

    gabarito[247 : 537, 1228 : 1518] = qr

def apply_student_info(gabarito, name, code, school):
    font = ImageFont.truetype("gerar_gabaritos/fonts/arial.ttf", 20)

    text_img = Image.new("RGB", (625, 25), color="white")
    draw_text = ImageDraw.Draw(text_img)
    draw_text.text((0, 0), name, (0, 0, 0), font)
    text = cv2.cvtColor((np.array(text_img)), cv2.COLOR_RGB2BGR)

    gabarito[458 : 483, 550 : 1175] = text

    text_img = Image.new("RGB", (625, 25), color="white")
    draw_text = ImageDraw.Draw(text_img)
    draw_text.text((0, 0), code, (0, 0, 0), font)
    cod = cv2.cvtColor(np.array(text_img), cv2.COLOR_RGB2BGR)

    gabarito[554 : 579, 526 : 1151] = cod

    text_img = Image.new("RGB", (625, 25), color="white")
    draw_text = ImageDraw.Draw(text_img)
    draw_text.text((0, 0), school, (0, 0, 0), font)
    school_name = cv2.cvtColor((np.array(text_img)), cv2.COLOR_RGB2BGR)

    gabarito[420 : 445, 520 : 1145] = school_name

def construir_gabaritos(parent=None):
    gabarito_alfa = cv2.imread("gerar_gabaritos/img/gabarito_alfa.png")
    gabarito_beta = cv2.imread("gerar_gabaritos/img/gabarito_beta.png")

    db = sqlite3.connect("database")
    cursor = db.cursor()
    cursor.execute("SELECT * FROM Aluno join Escola on Aluno.id_escola = Escola.id_escola")
    dados_alunos = cursor.fetchall()

    total = len(dados_alunos)

    def worker(cb, cancelled):
        for i, aluno in enumerate(dados_alunos, start=1):
            if cancelled["flag"]:
                break

            gab_alfa = gabarito_alfa.copy()
            gab_beta = gabarito_beta.copy()
            gabarito = gab_alfa if aluno[3] == "Alfa" else gab_beta
            escola = str(aluno[8])

            apply_qr_code(gabarito, aluno[2])
            apply_student_info(gabarito, aluno[1], aluno[2], escola.upper())


            file_path = f"gabaritos/{escola}/alfa" if aluno[3] == "Alfa" else f"gabaritos/{escola}/beta"

            if aluno[1] == "":
                file_path = file_path + "/extra"

            Path(file_path).mkdir(parents=True, exist_ok=True)
            
            cv2.imwrite(f"{file_path}/gabarito_{aluno[2]}.jpeg", gabarito, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

            cb(i, total)

    run_with_progress(parent, "Construindo gabaritos...", total, worker)

def obter_quant_gabaritos():
    db = sqlite3.connect("database")
    cursor = db.cursor()
    cursor.execute("SELECT * FROM Aluno")
    dados_alunos = cursor.fetchall()

    return len(dados_alunos)

"""
OFFSETS RETÂNGULO DO CANTO SUPERIOR DIREITO (LOCAL DO QRCODE):

1186 180
1560 180
1186 604
1560 604

"""