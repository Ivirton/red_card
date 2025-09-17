from PyQt6.QtWidgets import (
    QPushButton, QMessageBox, QWidget, 
    QVBoxLayout, QHBoxLayout, QTableView
)

from PyQt6.QtSql import QSqlTableModel

from utils import *

class TelaCrudAlunos(QWidget):
    def __init__(self, mainWindow):
        super().__init__()
        self.mainwindow = mainWindow
        layout = QVBoxLayout()
        self.db = connect_db()

        self.model = QSqlTableModel(self, self.db)
        self.model.setTable("Escola")
        self.model.setEditStrategy(QSqlTableModel.EditStrategy.OnManualSubmit)
        self.model.select()
        
        self._dirty = False
        self.model.dataChanged.connect(lambda *args: setattr(self, "_dirty", True))
        self.model.rowsInserted.connect(lambda *args: setattr(self, "_dirty", True))
        self.model.rowsRemoved.connect(lambda *args: setattr(self, "_dirty", True))

        self.view = QTableView()
        self.view.setModel(self.model)
        
        header = self.view.horizontalHeader()
        header.moveSection(0, 4)
        
        layout.addWidget(self.view)

        botoes_layout = QHBoxLayout()

        btn_salvar = QPushButton("Salvar")
        btn_salvar.clicked.connect(self.salvar_alteracoes)
        botoes_layout.addWidget(btn_salvar)

        btn_cancelar = QPushButton("Cancelar")
        btn_cancelar.clicked.connect(self.cancelar_alteracoes)
        botoes_layout.addWidget(btn_cancelar)

        btn_voltar = QPushButton("Voltar")
        btn_voltar.clicked.connect(self.voltar)  
        botoes_layout.addWidget(btn_voltar)

        layout.addLayout(botoes_layout)
        self.setLayout(layout)

    def salvar_alteracoes(self):
        if not self.model.submitAll():
            QMessageBox.critical(self, "Erro", f"Erro ao salvar: {self.model.lastError().text()}")
        else:
            self._dirty = False
            QMessageBox.information(self, "Alterações salvas.", "")

    def cancelar_alteracoes(self):
        self.model.revertAll()
        self._dirty = False

    def voltar(self):
        if self._dirty:
            msg = QMessageBox(self)
            msg.setWindowTitle("Aviso")
            msg.setText("Existem alterações não salvas. Deseja salvar antes de sair?")

            btn_salvar = msg.addButton("Salvar", QMessageBox.ButtonRole.AcceptRole)
            btn_descartar = msg.addButton("Descartar", QMessageBox.ButtonRole.DestructiveRole)
            btn_cancelar = msg.addButton("Cancelar", QMessageBox.ButtonRole.RejectRole)

            msg.exec()

            clicked = msg.clickedButton()
    
            if clicked == btn_salvar:
                self.salvar_alteracoes()
                self.mainwindow.exibir_tela_inicial()

            elif clicked == btn_descartar:
                self.cancelar_alteracoes()
                self.mainwindow.exibir_tela_inicial()
            
            return
        
        self.mainwindow.exibir_tela_inicial()