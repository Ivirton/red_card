import sys
from PyQt6.QtSql import QSqlDatabase, QSqlQuery

def connect_db():
    if QSqlDatabase.contains("qt_sql_default_connection"):
        db = QSqlDatabase.database("qt_sql_default_connection")

    else:
        db = QSqlDatabase.addDatabase("QSQLITE")
        db.setDatabaseName("database")
    
    if not db.isOpen():
        if not db.open():
            print("Erro ao conectar ao banco de dados!")
            return None
        
    return db

class DBAcess:
    def __init__(self, table_name):
        self.db = connect_db()
        self.table_name = table_name
    
    def execute_query(self, query_str):
        query = QSqlQuery(self.db)

        if not query.exec(query_str):
            print(f"Erro ao executar a query: {query.lastError().text()}")
            return None
        return query

    def get_all(self):
        query = self.execute_query(f"SELECT * FROM {self.table_name}")

        if not query:
            return []

        results = []

        while query.next():
            results.append(tuple(query.value(i) for i in range(query.record().count())))

        return results
    
class AlunoData(DBAcess):
    def __init__(self):
        super().__init__("Aluno")
    
    def get_by_escola(self, id_escola):
        query = QSqlQuery(self.db)
        query.prepare(f"SELECT * FROM {self.table_name} WHERE id_escola = :id_escola")
        query.bindValue(":id_escola", id_escola)

        if not query.exec():
            print(f"Erro ao buscar alunos da escola {id_escola}: {query.lastError().text()}")
            return []

        results = []

        while query.next():
            results.append(tuple(query.value(i) for i in range(query.record().count())))

        return results
    
    def insert(self, nome, codigo, nivel_prova, necessidade_especial, descricao_necessidade, id_escola):
        query = QSqlQuery(self.db)

        sql = f"""
            INSERT INTO {self.table_name} (nome, codigo, nivel_prova, necessidade_especial, descricao_necessidade, id_escola)
            VALUES (:nome, :codigo, :nivel_prova, :necessidade_especial, :descricao_necessidade, :id_escola)
        """

        query.prepare(sql)
        query.bindValue(":nome", nome)
        query.bindValue(":codigo", codigo)
        query.bindValue(":nivel_prova", nivel_prova)
        query.bindValue(":necessidade_especial", necessidade_especial)
        query.bindValue(":descricao_necessidade", descricao_necessidade)
        query.bindValue(":id_escola", id_escola)

        if not query.exec():
            print(f"Erro ao inserir escola: {query.lastError().text()}")
            return None  

        return query.lastInsertId()


    def update(self, id_aluno, nome, codigo, nivel_prova, necessidade_especial, descricao_necessidade, id_escola):
        query = QSqlQuery(self.db)

        sql = f"""
            UPDATE {self.table_name}
            SET nome = :nome,
                codigo = :codigo,
                nivel_prova = :nivel_prova,
                necessidade_especial = :necessidade_especial,
                descricao_necessidade = :descricao_necessidade, 
                id_escola = :id_escola
            WHERE id_aluno = :id
        """

        query.prepare(sql)
        query.bindValue(":nome", nome)
        query.bindValue(":codigo", codigo)
        query.bindValue(":nivel_prova", nivel_prova)
        query.bindValue(":necessidade_especial", necessidade_especial)
        query.bindValue(":id", id_aluno)
        query.bindValue(":descricao_necessidade", descricao_necessidade)
        query.bindValue(":id_escola", id_escola)

        if not query.exec():
            print(f"Erro ao atualizar aluno {id_aluno}: {query.lastError().text()}")
            return False

        return True

    def delete(self, id_aluno):
        query = QSqlQuery(self.db)

        sql = f"DELETE FROM {self.table_name} WHERE id_aluno = :id"

        query.prepare(sql)
        query.bindValue(":id", id_aluno)

        if not query.exec():
            print(f"Erro ao excluir aluno {id_aluno}: {query.lastError().text()}")
            return False

        return True

class EscolaData(DBAcess):
    def __init__(self):
        super().__init__("Escola")
    
    def insert(self, nome, email, inep, area):
        query = QSqlQuery(self.db)

        sql = f"""
            INSERT INTO {self.table_name} (nome, email, inep, area)
            VALUES (:nome, :email, :inep, :area)
        """

        query.prepare(sql)
        query.bindValue(":nome", nome)
        query.bindValue(":email", email)
        query.bindValue(":inep", inep)
        query.bindValue(":area", area)

        if not query.exec():
            print(f"Erro ao inserir escola: {query.lastError().text()}")
            return None  

        return query.lastInsertId()

    def update(self, id_escola, nome, email, inep, area):
        query = QSqlQuery(self.db)

        sql = f"""
            UPDATE {self.table_name}
            SET nome = :nome,
                email = :email,
                inep = :inep,
                area = :area
            WHERE id_escola = :id
        """

        query.prepare(sql)
        query.bindValue(":nome", nome)
        query.bindValue(":email", email)
        query.bindValue(":inep", inep)
        query.bindValue(":area", area)
        query.bindValue(":id", id_escola)

        if not query.exec():
            print(f"Erro ao atualizar escola {id_escola}: {query.lastError().text()}")
            return False

        return True

    def delete(self, escola_id):
        query = QSqlQuery(self.db)

        sql = f"DELETE FROM {self.table_name} WHERE id_escola = :id"

        query.prepare(sql)
        query.bindValue(":id", escola_id)

        if not query.exec():
            print(f"Erro ao excluir escola {escola_id}: {query.lastError().text()}")
            return False

        return True