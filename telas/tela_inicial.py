from PyQt6.QtWidgets import QPushButton, QMessageBox, QWidget, QVBoxLayout, QLabel

from gerar_gabaritos.construir_gabaritos import construir_gabaritos, obter_quant_gabaritos
from gerar_gabaritos.gerar_gabaritos import gerar_gabaritos

from utils import *

class TelaInicial(QWidget):
    def __init__(self, mainwindow):
        super().__init__()
        self.mainwindow = mainwindow

        layout = QVBoxLayout()

        label = QLabel("Escolha uma das opções abaixo:")
        layout.addWidget(label)

        btn1 = QPushButton("1. Visualizar Alunos Cadastrados")
        btn1.clicked.connect(self.mainwindow.exibir_tela_crud_escolas)
        layout.addWidget(btn1)

        btn2 = QPushButton("2. Gerar Gabaritos")
        btn2.clicked.connect(self.gerar_gabaritos)
        layout.addWidget(btn2)

        btn3 = QPushButton("3. Corrigir Provas")
        btn3.clicked.connect(self.mainwindow.exibir_tela_correcao)
        layout.addWidget(btn3)

        btn_sair = QPushButton("4. Sair")
        btn_sair.clicked.connect(self.mainwindow.close)
        layout.addWidget(btn_sair)

        self.setLayout(layout)

    def gerar_gabaritos(self):
        if dialogo_confirmacao(parent=None, titulo="", mensagem=f"Deseja mesmo gerar {obter_quant_gabaritos()} gabaritos?"):
            construir_gabaritos(self)
            gerar_gabaritos(self)
            deletar_gabaritos_temp()
            QMessageBox.information(self, "Gabaritos gerados!", "Os gabaritos foram salvos na pasta 'gabaritos_pdf'.")