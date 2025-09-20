from PyQt6.QtWidgets import QComboBox, QLineEdit, QStyledItemDelegate, QStyle
from PyQt6.QtGui import QPainter

class NoScrollComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()

from PyQt6.QtWidgets import QStyledItemDelegate, QLineEdit, QStyle
from PyQt6.QtGui import QPainter

class CleanEditDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)
        editor.setFrame(False)
        return editor

    def setEditorData(self, editor, index):
        value = index.data()
        editor.setText(str(value))
        editor.selectAll()

    def setModelData(self, editor, model, index):
        value = editor.text()
        model.setData(index, value)

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)

    def paint(self, painter: QPainter, option, index):
        table = index.model().parent()
        if table and table.indexWidget(index):  
            painter.save()
            painter.fillRect(option.rect, option.palette.base())
            painter.restore()
            return
        
        super().paint(painter, option, index)
