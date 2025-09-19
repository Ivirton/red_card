from PyQt6.QtWidgets import (
    QPushButton, QMessageBox, QWidget, QVBoxLayout, 
    QHBoxLayout, QHeaderView, QTableWidget, 
    QTableWidgetItem, QDialog, QFormLayout, QLineEdit
)

from utils import *
from telas.telas_utils import *

import db.db_crud as db_acess

class DialogAdicionarAluno(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Adicionar Aluno")
        self.setModal(True)

        layout = QFormLayout()

        self.input_nome = QLineEdit()
        layout.addRow("Nome:", self.input_nome)

        self.input_email = QLineEdit()
        layout.addRow("E-mail:", self.input_email)

        self.input_inep = QLineEdit()
        layout.addRow("Inep:", self.input_inep)

        self.combo_area = NoScrollComboBox()
        self.combo_area.addItems(["Sede", "Campo"])
        layout.addRow("Área:", self.combo_area)

        btn_layout = QHBoxLayout()
        self.btn_ok = QPushButton("Adicionar")
        self.btn_cancel = QPushButton("Cancelar")
        btn_layout.addWidget(self.btn_ok)
        btn_layout.addWidget(self.btn_cancel)

        v_layout = QVBoxLayout()
        v_layout.addLayout(layout)
        v_layout.addLayout(btn_layout)
        self.setLayout(v_layout)

        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok.clicked.connect(self.accept)

    def get_data(self):
        return {
            "nome": self.input_nome.text().strip(),
            "email": self.input_email.text().strip(),
            "inep": self.input_inep.text().strip(),
            "area": self.combo_area.currentText()
        }

class TelaCrudAlunos(QWidget):
    def __init__(self, mainWindow, id_escola):
        super().__init__()
        self.mainwindow = mainWindow
        self.id_escola = id_escola
        self._dirty = False
        self.table = QTableWidget()
        self.delete_rows = set()

        layout = QVBoxLayout()

        self.carregar_dados()

        self.table.setColumnHidden(0, True)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.itemChanged.connect(self.item_changed) 

        self.table.setItemDelegate(CleanEditDelegate())

        layout.addWidget(self.table)

        botoes_layout = QHBoxLayout()

        btn_add_escola = QPushButton("Adicionar Aluno")
        # btn_add_escola.clicked.connect(self.adicionar_aluno)
        botoes_layout.addWidget(btn_add_escola)

        btn_salvar = QPushButton("Salvar")
        # btn_salvar.clicked.connect(self.salvar_alteracoes)
        botoes_layout.addWidget(btn_salvar)

        btn_cancelar = QPushButton("Cancelar Alterações Atuais")
        # btn_cancelar.clicked.connect(self.cancelar_alteracoes)
        botoes_layout.addWidget(btn_cancelar)

        btn_voltar = QPushButton("Voltar")
        btn_voltar.clicked.connect(self.voltar)
        botoes_layout.addWidget(btn_voltar)

        layout.addLayout(botoes_layout)
        self.setLayout(layout)
    
    def carregar_dados(self):
        self.table.blockSignals(True)

        alunos = db_acess.AlunoData().get_by_escola(self.id_escola)
        self.original_data = [list(escola) for escola in alunos] 

        self.table.setRowCount(len(alunos))
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "id_aluno", "Nome do Aluno", "Código", "Nível Prova", "Necessidade Especial", 
            "Descrição Necessidade Especial", "Excluir Aluno"
        ])

        for row_index, aluno in enumerate(alunos):
            for i in [0, 1, 2, 5]:
                item = QTableWidgetItem(str(aluno[i]))

                if i == 2:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)

                self.table.setItem(row_index, i, item)
            
            combo1 = NoScrollComboBox()
            combo1.addItems(["Alfa", "Beta"])
            combo1.setCurrentText(str(aluno[3]))
            combo1.currentIndexChanged.connect(lambda _, r=row_index: self.combo_changed(r))
            self.table.setCellWidget(row_index, 3, combo1)

            combo2 = NoScrollComboBox()
            combo2.addItems(["Sim", "Não"])
            combo2.setCurrentText(str(aluno[4]))
            combo2.currentIndexChanged.connect(lambda _, r=row_index: self.combo_changed(r))
            self.table.setCellWidget(row_index, 4, combo2)

            btn2 = QPushButton("Excluir")
            btn2.clicked.connect(lambda checked, r=row_index: self.excluir_aluno(r))
            self.table.setCellWidget(row_index, 6, btn2)

        self.table.blockSignals(False)

    def cancelar_alteracoes(self):
        self.table.blockSignals(True)

        for row in range(self.table.rowCount()):
            if row < len(self.original_data):
                for col in range(0, 4):
                    self.table.item(row, col).setText(str(self.original_data[row][col]))
                
                combo = self.table.cellWidget(row, 4)

                if combo:
                    combo.setCurrentText(str(self.original_data[row][4]))
            
            item_id = int(self.table.item(row, 0).text())

            if item_id in self.delete_rows:
                self.table.showRow(row)
    
        self.delete_rows.clear()

        self._dirty = False
        self.table.blockSignals(False)

    def salvar_alteracoes(self):
        msg = QMessageBox(self) 
        msg.setWindowTitle("Aviso") 
        msg.setText("Tem certeza que deseja salvar as alterações? Não poderão ser desfeitas depois!") 
        
        btn_salvar = msg.addButton("Salvar", QMessageBox.ButtonRole.AcceptRole) 
        btn_cancelar = msg.addButton("Cancelar", QMessageBox.ButtonRole.RejectRole) 
        
        msg.exec() 
        
        clicked = msg.clickedButton() 
        
        if clicked == btn_salvar: 
            try:
                escola_data = db_acess.EscolaData()  
                db = escola_data.db
                db.transaction()

                for escola_id in self.delete_rows:
                    escola_data.delete(escola_id)

                self.delete_rows.clear()

                for row in range(self.table.rowCount()):
                    escola_id_item = self.table.item(row, 0)
                    escola_id = escola_id_item.text().strip()
                    nome = self.table.item(row, 1).text()
                    email = self.table.item(row, 2).text()
                    inep = self.table.item(row, 3).text()
                    area = self.table.cellWidget(row, 4).currentText()

                    if escola_id == "":
                        nova_id = escola_data.insert(nome=nome, email=email, inep=inep, area=area)
                        escola_id_item.setText(str(nova_id)) 
                        self.original_data.append([nova_id, nome, email, inep, area])

                    else:
                        escola_data.update(escola_id=int(escola_id), nome=nome, email=email, inep=inep, area=area)
                        index = row
                        self.original_data[index] = [int(escola_id), nome, email, inep, area]

                db.commit()
                self._dirty = False
                QMessageBox.information(self, "Sucesso", "Alterações salvas com sucesso!")

            except Exception as e:
                db.rollback()
                QMessageBox.critical(self, "Erro", f"Erro ao salvar: {e}")
            
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
                # self.salvar_alteracoes()
                self.mainwindow.exibir_tela_crud_escolas()

            elif clicked == btn_descartar:
                # self.cancelar_alteracoes()
                self.mainwindow.exibir_tela_crud_escolas()
            
            return  
        
        self.mainwindow.exibir_tela_crud_escolas()

    def item_changed(self, item):
        row = item.row()
        col = item.column()

        if row < len(self.original_data):
            if col >= 4:
                return

            novo_valor = item.text()
            valor_original = str(self.original_data[row][col])

            if novo_valor != valor_original:
                self._dirty = True
    
    def combo_changed(self, row):
        if row < len(self.original_data):
            novo_valor = self.table.cellWidget(row, 4).currentText()
            valor_original = str(self.original_data[row][4])

            if novo_valor != valor_original:
                self._dirty = True
    
    def adicionar_aluno(self):
        dialog = DialogAdicionarAluno(self)

        if dialog.exec():
            data = dialog.get_data()
            
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(""))
            self.table.setItem(row, 1, QTableWidgetItem(data["nome"]))
            self.table.setItem(row, 2, QTableWidgetItem(data["email"]))
            self.table.setItem(row, 3, QTableWidgetItem(data["inep"]))

            combo = NoScrollComboBox()
            combo.addItems(["Sede", "Campo"])
            combo.setCurrentText(data["area"])
            combo.currentIndexChanged.connect(lambda _, r=row: self.combo_changed(r))
            self.table.setCellWidget(row, 4, combo)

            btn1 = QPushButton("Ver Alunos")
            btn1.clicked.connect(lambda _, r=row: self.abrir_alunos(r))
            self.table.setCellWidget(row, 5, btn1)

            btn2 = QPushButton("Excluir")
            btn2.clicked.connect(lambda _, r=row: self.excluir_escola(r))
            self.table.setCellWidget(row, 6, btn2)

            self._dirty = True
            QMessageBox.information(self, "Sucesso", "Escola adicionada com sucesso!")

        
    def excluir_aluno(self, row):
        nome_escola = str(self.table.item(row, 1).text())
        
        msg = QMessageBox(self) 
        msg.setWindowTitle("Aviso") 
        msg.setText(f"Tem certeza que deseja remover a escola \"{nome_escola}\"? Essa ação não poderá ser desfeita!") 
            
        btn_excluir = msg.addButton("Excluir", QMessageBox.ButtonRole.AcceptRole) 
        btn_cancelar = msg.addButton("Cancelar", QMessageBox.ButtonRole.RejectRole) 
            
        msg.exec() 
            
        clicked = msg.clickedButton() 
            
        if clicked == btn_excluir: 
            try:
                item_id_text = self.table.item(row, 0).text()

                if item_id_text.isdigit():
                    self.delete_rows.add(int(item_id_text))
                    self.table.hideRow(row) 

                else:
                    self.table.removeRow(row)   
            
                self._dirty = True
                QMessageBox.information(self, "Sucesso", f"A escola {nome_escola} foi removida com sucesso!")

            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Erro ao salvar: {e}")