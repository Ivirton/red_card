import sys

from PyQt6.QtWidgets import QApplication, QMainWindow, QStackedWidget

from telas.tela_correcao import TelaCorrecao
from telas.tela_inicial import TelaInicial
from telas.tela_crud_escolas import TelaCrudEscolas
from telas.tela_crud_alunos import TelaCrudAlunos

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OLIMPÍADA CAXIENSE DE MATEMÁTICA DAS ESCOLAS MUNICIPAIS")
        self.setGeometry(200, 200, 900, 600)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.tela_inicial = TelaInicial(self)
        self.tela_correcao = TelaCorrecao(self)
        self.tela_crud_escolas = TelaCrudEscolas(self)

        self.stack.addWidget(self.tela_inicial)
        self.stack.addWidget(self.tela_correcao)
        self.stack.addWidget(self.tela_crud_escolas)

        self.stack.setCurrentWidget(self.tela_inicial)

    def exibir_tela_inicial(self):
        self.stack.setCurrentWidget(self.tela_inicial)
        self.setWindowTitle("OLIMPÍADA CAXIENSE DE MATEMÁTICA DAS ESCOLAS MUNICIPAIS")

    def exibir_tela_correcao(self):
        self.stack.setCurrentWidget(self.tela_correcao)
        self.setWindowTitle("OLIMPÍADA CAXIENSE DE MATEMÁTICA DAS ESCOLAS MUNICIPAIS - CORREÇÃO DE PROVAS")

    def exibir_tela_crud_escolas(self):
        self.stack.setCurrentWidget(self.tela_crud_escolas)
        self.setWindowTitle("OLIMPÍADA CAXIENSE DE MATEMÁTICA DAS ESCOLAS MUNICIPAIS - VISUALIZAR ESCOLAS")

    def exibir_tela_alunos(self, escola_id, nome_escola):
        self.tela_crud_alunos = TelaCrudAlunos(self, escola_id)
        self.stack.addWidget(self.tela_crud_alunos)
        self.stack.setCurrentWidget(self.tela_crud_alunos)
        self.setWindowTitle(f"VISUALIZANDO ALUNOS - {nome_escola}")
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())