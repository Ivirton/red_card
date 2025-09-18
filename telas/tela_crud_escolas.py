from PyQt6.QtCore import (
    pyqtSignal, QModelIndex, QModelIndex, pyqtSignal,
    QIdentityProxyModel, QSize, Qt
)

from PyQt6.QtWidgets import (
    QPushButton, QMessageBox, QWidget, QVBoxLayout, 
    QHBoxLayout, QTableView,  QStyledItemDelegate, 
    QStyle, QStyleOptionButton, QApplication, QLineEdit,
    QComboBox, QHeaderView
)

from PyQt6.QtSql import QSqlTableModel
from PyQt6.QtGui import QIntValidator

from utils import *

class ProxyComBotoes(QIdentityProxyModel):
    def columnCount(self, parent=QModelIndex()):
        return super().columnCount(parent) + 1  

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.column() == self.columnCount() - 1: 
            if role == Qt.ItemDataRole.DisplayRole:
                return ""
            elif role == Qt.ItemDataRole.SizeHintRole:
                return QSize(100, 30)
        return super().data(index, role)

    def flags(self, index):
        if index.column() == self.columnCount() - 1:
            return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        return super().flags(index)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            if section == self.columnCount() - 1:
                return "Ações"
        return super().headerData(section, orientation, role)

class ButtonDelegate(QStyledItemDelegate):
    clicked = pyqtSignal(QModelIndex)  

    def __init__(self, parent=None):
        super().__init__(parent)
        self._buttons = {}

    def paint(self, painter, option, index):
        btn_option = QStyleOptionButton()
        btn_option.rect = option.rect.adjusted(4, 4, -4, -4) 
        btn_option.text = "Ver Alunos"
        btn_option.state = QStyle.StateFlag.State_Enabled | QStyle.StateFlag.State_Raised
        QApplication.style().drawControl(QStyle.ControlElement.CE_PushButton, btn_option, painter)

        self._buttons[(index.row(), index.column())] = btn_option.rect

    def editorEvent(self, event, model, option, index):
       key = (index.row(), index.column())
       
       if key in self._buttons and self._buttons[key].contains(event.pos()):
            self.clicked.emit(index)
            return True
       return False

class CustomDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        col = index.column()

        if col in [1, 2, 3]:
            editor = QLineEdit(parent)
            return editor
        
        elif col == 4:
            editor = QComboBox(parent)
            editor.addItems(["Sede", "Campo"])
            return editor

        return super().createEditor(parent, option, index)

class TelaCrudEscolas(QWidget):
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

        self.proxy = ProxyComBotoes(self)
        self.proxy.setSourceModel(self.model)
        self.view.setModel(self.proxy)

        self.view.setColumnHidden(0, True)

        edit_delegate = CustomDelegate(self.view)
        self.view.setItemDelegate(edit_delegate)

        delegate = ButtonDelegate(self.view)
        self.view.setItemDelegateForColumn(self.proxy.columnCount() - 1, delegate)
        self.view.setColumnWidth(self.proxy.columnCount() - 1, 120)  

        def abrir_alunos(index):
            src_index = self.proxy.mapToSource(index)
            record = self.model.record(src_index.row())
            escola_id = record.value("id")
            # self.mainwindow.exibir_tela_alunos(escola_id)

        delegate.clicked.connect(abrir_alunos)

        self.view.horizontalHeader().setStretchLastSection(False)
        self.view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        layout.addWidget(self.view)

        botoes_layout = QHBoxLayout()

        btn_add_escola = QPushButton("Adicionar Escola")
        # btn_add_escola.clicked.connect()
        botoes_layout.addWidget(btn_add_escola)

        btn_salvar = QPushButton("Salvar")
        btn_salvar.clicked.connect(self.salvar_alteracoes)
        botoes_layout.addWidget(btn_salvar)

        btn_cancelar = QPushButton("Cancelar Alterações Atuais")
        btn_cancelar.clicked.connect(self.cancelar_alteracoes)
        botoes_layout.addWidget(btn_cancelar)

        btn_voltar = QPushButton("Voltar")
        btn_voltar.clicked.connect(self.voltar)
        botoes_layout.addWidget(btn_voltar)

        layout.addLayout(botoes_layout)
        self.setLayout(layout)

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

    def salvar_alteracoes(self):
        msg = QMessageBox(self) 
        msg.setWindowTitle("Aviso") 
        msg.setText("Tem certeza que deseja salvar as alterações? Não poderão ser desfeitas depois!") 
        
        btn_salvar = msg.addButton("Salvar", QMessageBox.ButtonRole.AcceptRole) 
        btn_cancelar = msg.addButton("Cancelar", QMessageBox.ButtonRole.RejectRole) 
        
        msg.exec() 
        
        clicked = msg.clickedButton() 
        
        if clicked == btn_salvar: 
            if not self.model.submitAll(): 
                QMessageBox.critical(self, "Erro", f"Erro ao salvar: {self.model.lastError().text()}") 
                return 
            
            self._dirty = False 
            QMessageBox.information(self, "Alterações salvas.", "Todas as alterações foram salvas.")

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