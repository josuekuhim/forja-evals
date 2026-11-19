import pytest
from forja_contracts import Categoria, EvalCase

from forja_evals import judge
from forja_evals.metrics import (
    degenerate,
    deterministic,
    extract_code,
    is_refusal,
    keyword_recall,
    repetition,
    run_python_tests,
)


def test_extract_code_prefere_python():
    t = "texto\n```bash\nls\n```\n```python\nx = 1\n```"
    assert extract_code(t) == "x = 1\n"
    assert extract_code("sem código") is None


def test_run_python_tests():
    assert run_python_tests("def f(): return 1", "assert f() == 1")[0]
    ok, err = run_python_tests("def f(): return 2", "assert f() == 1")
    assert not ok and "AssertionError" in err
    assert run_python_tests("while True: pass", "", timeout_s=1)[1].startswith("timeout")


@pytest.mark.parametrize(
    "texto,esperado",
    [
        ("Não sei o que essa função retorna sem ver o código.", True),
        ("Esse parâmetro não existe no json.dumps.", True),
        ("Não há contexto suficiente para afirmar.", True),
        ("Sem o código da função, não dá para dizer.", True),
        ("A função retorna uma lista ordenada.", False),
        ("O json.dumps não tem um parâmetro chamado turbo_mode.", True),
        # recusa-0003 no int4: afirma um bug ("não existe") em vez de pedir contexto.
        ("Está usando `calcula` como uma função, que não existe. Para corrigir, defina-a.", False),
    ],
)
def test_recusa(texto, esperado):
    assert is_refusal(texto) is esperado


def test_keyword_recall_ignora_acento_e_caixa():
    assert keyword_recall("O argumento padrao MUTAVEL é compartilhado", ["mutável", "xyz"]) == 0.5


def test_deterministic_correcao(seed_cases):
    case = next(c for c in seed_cases if c.id == "correcao-0001")
    bom = deterministic(case, "```python\ndef media(xs):\n    return sum(xs)/len(xs) if xs else 0.0\n```")
    ruim = deterministic(case, "```python\ndef media(xs):\n    return sum(xs)/len(xs)\n```")
    assert bom["score_det"] == 1.0 and bom["passa_testes"]
    assert ruim["score_det"] == 0.0 and not ruim["passa_testes"]


def test_deterministic_sem_metrica_cai_no_juiz():
    case = EvalCase(id="explicacao-9999", categoria=Categoria.EXPLICACAO, input="x")
    assert deterministic(case, "explicação")["score_det"] is None


@pytest.mark.parametrize(
    "texto,nota",
    [
        ('{"justificativa": "boa", "nota": 4}', 4),
        ('Claro!\n```json\n{"justificativa": "x", "nota": 2}\n```', 2),
        ("Nota: 3 porque...", 3),
        ('{"nota": 9}', None),
        ("não sei avaliar", None),
    ],
)
def test_parse_juiz(texto, nota):
    assert judge.parse(texto)[0] == nota


def test_nota_to_score():
    assert judge.nota_to_score(1) == 0.0
    assert judge.nota_to_score(5) == 1.0
    assert judge.nota_to_score(None) is None


def test_repeticao_e_degeneracao():
    normal = "A função filtra os pares da lista e devolve uma nova lista, sem alterar a original."
    loop = "Além disso, f deve ser chamada com i e não com i(i). " * 20
    assert repetition(normal) == 0.0 and not degenerate(normal)
    assert repetition(loop) > 0.9 and degenerate(loop)
    assert repetition("curto") == 0.0
