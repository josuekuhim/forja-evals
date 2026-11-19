"""Métricas determinísticas. Quando uma categoria tem métrica determinística, ela é a nota principal;
o juiz só decide onde não há como verificar automaticamente."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

from forja_contracts import Categoria, EvalCase

CODE_BLOCK = re.compile(r"```([\w+-]*)[ \t]*\r?\n(.*?)```", re.S)

REFUSAL_PATTERNS = [
    r"\bnao sei\b",
    r"\bnao (tenho|ha|temos) (informac\w+|contexto|dados) suficientes?\b",
    r"\bnao (e|seria) possivel (determinar|saber|afirmar|dizer|responder)\b",
    r"\bnao (consigo|da para|da pra|posso) (determinar|afirmar|saber|dizer|responder)\b",
    r"\bsem (mais |o )?(contexto|codigo|acesso)\b",
    # "não existe" só vale para parâmetro/opção inventado (recusa-0002). Solto, casava com
    # "usa `calcula`, que não existe", que é o modelo AFIRMANDO um bug, não recusando.
    r"\b(parametro|argumento|opcao|flag|atributo)\b[^.\n]{0,80}\bnao existe\b",
    r"\bnao (existe|ha|tem) (nenhum |nenhuma |esse |essa |um |uma |o |a |tal )?"
    r"(parametro|argumento|opcao|flag|atributo)\b",
    r"\bnao (aceita|suporta|possui) (um |o |esse |nenhum |tal )?(parametro|argumento|opcao)\b",
    r"\bnao tenho (acesso|certeza)\b",
    r"\bpreciso (de mais|ver o|do codigo)\b",
]
_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS))


# Degeneração: fração de 8-gramas de palavras repetidos. Calibrado nas respostas reais: respostas
# saudáveis ficam ≤ 0,16 (listas de exemplos repetem estrutura); loops ficam entre 0,36 e 0,93.
REPETITION_N = 8
DEGENERATION_THRESHOLD = 0.3


def repetition(text: str, n: int = REPETITION_N) -> float:
    """0 = nenhum n-grama repetido; perto de 1 = o texto é a mesma frase em loop."""
    words = re.findall(r"\w+", text.casefold())
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 0.0 if not grams else 1 - len(set(grams)) / len(grams)


def degenerate(text: str) -> bool:
    return repetition(text) >= DEGENERATION_THRESHOLD


def normalize(text: str) -> str:
    """Minúsculas e sem acento, para casar palavras-chave e padrões em português."""
    nfkd = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def extract_code(text: str, lang: str = "python") -> str | None:
    blocks = CODE_BLOCK.findall(text)
    if not blocks:
        return None
    for tag, body in blocks:
        if tag.lower() in (lang, lang[:2]):
            return body
    return blocks[0][1]


def compiles(code: str) -> bool:
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def run_python_tests(code: str, tests: str, timeout_s: float = 10.0) -> tuple[bool, str]:
    """Roda código + asserts num subprocesso isolado (`python -I`, diretório temporário, timeout).

    ATENÇÃO: isso NÃO é um sandbox. Executa código gerado por modelo. No CI, rode dentro de um
    container descartável sem segredos e sem rede.
    """
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "solucao.py"
        f.write_text(code + "\n\n# --- testes ---\n" + tests + "\n", encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(f)], cwd=d, capture_output=True, text=True, timeout=timeout_s
            )
        except subprocess.TimeoutExpired:
            return False, f"timeout ({timeout_s}s)"
    return proc.returncode == 0, proc.stderr[-500:]


def is_refusal(text: str) -> bool:
    return bool(_REFUSAL_RE.search(normalize(text)))


def keyword_recall(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    t = normalize(text)
    return sum(normalize(k) in t for k in keywords) / len(keywords)


def deterministic(case: EvalCase, output: str, *, timeout_s: float = 10.0) -> dict[str, Any]:
    """Retorna métricas e `score_det` (0..1) quando a categoria permite verificação automática."""
    rep = repetition(output)
    m: dict[str, Any] = {
        "recusou": is_refusal(output),
        "repeticao": round(rep, 3),
        "degenerou": rep >= DEGENERATION_THRESHOLD,
        "score_det": None,
    }
    meta = case.metadados

    if case.categoria == Categoria.RECUSA:
        m["score_det"] = 1.0 if m["recusou"] else 0.0
        return m

    if case.categoria == Categoria.CORRECAO and meta.get("testes"):
        code = extract_code(output, meta.get("linguagem", "python"))
        m["tem_codigo"] = code is not None
        m["compila"] = bool(code) and compiles(code)
        if m["compila"]:
            passed, detail = run_python_tests(code, meta["testes"], timeout_s)
            m["passa_testes"] = passed
            if not passed:
                m["erro_testes"] = detail
        else:
            m["passa_testes"] = False
        m["score_det"] = 1.0 if m["passa_testes"] else 0.0
        return m

    if case.categoria == Categoria.DETECCAO_BUG and meta.get("palavras_chave"):
        m["recall_palavras_chave"] = keyword_recall(output, meta["palavras_chave"])
        m["score_det"] = 1.0 if m["recall_palavras_chave"] > 0 else 0.0
    return m
