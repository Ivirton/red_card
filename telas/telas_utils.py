from PyQt6.QtWidgets import QComboBox, QLineEdit, QStyledItemDelegate, QStyle, QStyleOptionViewItem

class NoScrollComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()

class CleanEditDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)  
        editor.setFrame(False)       
        editor.setText("")           
        return editor                

    def setEditorData(self, editor, index):
        value = index.data()         
        editor.setText(str(value))   

    def setModelData(self, editor, model, index):
        value = editor.text()        
        model.setData(index, value)  

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)
    
    def paint(self, painter, option: QStyleOptionViewItem, index):
        if option.state & QStyle.StateFlag.State_Editing:
            painter.save()
            painter.fillRect(option.rect, option.palette.base())
            painter.restore()
            return
        super().paint(painter, option, index)