import sqlite3

# Conectar a um banco em memória (pode trocar para um arquivo .db se quiser salvar)
conn = sqlite3.connect("database")
cursor = conn.cursor()

# Ativar foreign keys
cursor.execute("PRAGMA foreign_keys = ON;")

# Script SQL na ordem correta
sql_script = """
CREATE TABLE IF NOT EXISTS escola (
    id_inep INTEGER PRIMARY KEY,
    nome TEXT NOT NULL UNIQUE,
    area TEXT CHECK(area IN ('Sede','Campo')),
    email TEXT,
    telefone TEXT
);

CREATE TABLE IF NOT EXISTS professor (
    id_prof INTEGER PRIMARY KEY AUTOINCREMENT,
    nome TEXT NOT NULL,
    id_escola INTEGER NOT NULL,
    FOREIGN KEY (id_escola) REFERENCES escola(id_inep)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS responsavel (
    id_resp INTEGER PRIMARY KEY AUTOINCREMENT,
    nome TEXT NOT NULL,
    email TEXT,
    id_escola INTEGER NOT NULL,
    FOREIGN KEY (id_escola) REFERENCES escola(id_inep)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS aluno (
    id_aluno INTEGER PRIMARY KEY AUTOINCREMENT,
    codigo_aluno TEXT NOT NULL UNIQUE,
    nome TEXT NOT NULL,
    cpf TEXT UNIQUE,
    uuid TEXT UNIQUE DEFAULT (lower(hex(randomblob(16)))),
    nivel TEXT NOT NULL CHECK(nivel IN ('Alfa','Beta')),
    necessidades_especiais INTEGER NOT NULL DEFAULT 0 CHECK(necessidades_especiais IN (0,1)),
    detalhes_necessidades TEXT,
    id_escola INTEGER,
    FOREIGN KEY(id_escola) REFERENCES escola(id_inep)
        ON DELETE SET NULL
        ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS prova (
    id_prova INTEGER PRIMARY KEY AUTOINCREMENT,
    nome TEXT NOT NULL,
    data_prova DATE NOT NULL,
    descricao TEXT
);

CREATE TABLE IF NOT EXISTS questao (
    id_questao INTEGER PRIMARY KEY AUTOINCREMENT,
    id_prova INTEGER NOT NULL,
    numero INTEGER NOT NULL,
    peso REAL NOT NULL DEFAULT 1.0,
    enunciado TEXT,
    FOREIGN KEY (id_prova) REFERENCES prova(id_prova)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    UNIQUE(id_prova, numero)
);

CREATE TABLE IF NOT EXISTS gabarito (
    id_gabarito INTEGER PRIMARY KEY AUTOINCREMENT,
    id_questao INTEGER NOT NULL UNIQUE,
    alternativa_correta TEXT NOT NULL,
    FOREIGN KEY (id_questao) REFERENCES questao(id_questao)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS cartao_resposta (
    id_cartao INTEGER PRIMARY KEY AUTOINCREMENT,
    id_prova INTEGER NOT NULL,
    id_aluno INTEGER NOT NULL,
    codigo_aluno_snapshot TEXT NOT NULL,
    cpf_snapshot TEXT,
    id_escola_snapshot INTEGER,
    pontuacao_total REAL DEFAULT 0,
    status TEXT CHECK(status IN ('Completo','Incompleto','Anulado')),
    FOREIGN KEY (id_prova) REFERENCES prova(id_prova)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    FOREIGN KEY (id_aluno) REFERENCES aluno(id_aluno)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS resposta (
    id_resposta INTEGER PRIMARY KEY AUTOINCREMENT,
    id_cartao INTEGER NOT NULL,
    id_questao INTEGER NOT NULL,
    alternativa TEXT,
    acertou INTEGER NOT NULL CHECK(acertou IN (0,1)),
    pontos_obtidos REAL NOT NULL DEFAULT 0,
    FOREIGN KEY (id_cartao) REFERENCES cartao_resposta(id_cartao)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    FOREIGN KEY (id_questao) REFERENCES questao(id_questao)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    UNIQUE(id_cartao, id_questao)
);
"""

# Executar script
cursor.executescript(sql_script)

# Listar tabelas criadas
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tabelas = cursor.fetchall()

tabelas
