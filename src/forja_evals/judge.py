"""LLM-as-judge com rubrica. A nota do juiz só vale depois de calibrada contra rótulos humanos (kappa)."""

from __future__ import annotations

import json
import re

from forja_contracts import EvalCase

from forja_evals.prompts import JUDGE_SYSTEM, RUBRICA_GERAL, RUBRICA_POR_CATEGORIA

_JSON_RE = re.compile(r"\{.*?\}", re.S)
_NOTA_RE = re.compile(r"nota\D{0,5}([1-5])", re.I)


def build_messages(case: EvalCase, output: str) -> list[dict[str, str]]:
    ref = f"\n\n## Resposta de referência\n{case.referencia}" if case.referencia else ""
    user = (
        f"## Rubrica\n{RUBRICA_GERAL}\n{RUBRICA_POR_CATEGORIA[case.categoria.value]}\n\n"
        f"## Pergunta do usuário\n{case.input}{ref}\n\n"
        f"## Resposta do assistente\n{output}"
    )
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def parse(text: str) -> tuple[int | None, str]:
    for m in _JSON_RE.finditer(text):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        nota = obj.get("nota")
        if isinstance(nota, int | float) and 1 <= int(nota) <= 5:
            return int(nota), str(obj.get("justificativa", ""))
    if m := _NOTA_RE.search(text):
        return int(m.group(1)), text.strip()[:300]
    return None, text.strip()[:300]


def nota_to_score(nota: int | None) -> float | None:
    return None if nota is None else (nota - 1) / 4
