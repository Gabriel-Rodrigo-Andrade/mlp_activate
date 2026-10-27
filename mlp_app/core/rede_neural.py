from collections.abc import Sequence

import numpy as np

from mlp_app.core.camada import Camada, FuncaoAtivacao, Valores


class RedeNeural:
    """Executa uma sequência de camadas densas."""

    def __init__(self, camadas: Sequence[Camada]) -> None:
        if not camadas:
            raise ValueError("A rede precisa ter ao menos uma camada.")
        
        self._camadas = tuple(camadas)

    def propagar(
        self,
        entradas: Valores,
        ativacoes: Sequence[FuncaoAtivacao | None] | None = None,
    ) -> np.ndarray:
        """Aplica cada camada e sua ativação opcional, na ordem definida."""

        if ativacoes is None:
            ativacoes = [None] * len(self._camadas)

        if len(ativacoes) != len(self._camadas):
            raise ValueError("Informe uma ativação para cada camada.")

        resultado = entradas

        for camada, ativacao in zip(self._camadas, ativacoes):
            resultado = camada.calcular(resultado, ativacao)

        return resultado
