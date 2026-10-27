from collections.abc import Callable
from typing import Union

import numpy as np

Valores = Union[list[float], list[list[float]], np.ndarray]
FuncaoAtivacao = Callable[[np.ndarray], np.ndarray]


class Camada:
    """Calcula entradas X pesos transpostos + bias."""

    def __init__(self, pesos: list[list[float]], bias: list[float]) -> None:
        self._pesos = np.asarray(pesos, dtype=float)
        self._bias = np.asarray(bias, dtype=float)

        if self._pesos.ndim != 2 or 0 in self._pesos.shape:
            raise ValueError("pesos deve ser uma matriz não vazia.")
        
        if self._bias.ndim != 1 or self._bias.size != self._pesos.shape[0]:
            raise ValueError("Deve existir um bias para cada linha de pesos.")
        
        if not np.isfinite(self._pesos).all() or not np.isfinite(self._bias).all():
            raise ValueError("Pesos e bias devem conter números finitos.")

    def calcular(
        self,
        entradas: Valores,
        ativacao: FuncaoAtivacao | None = None,
    ) -> np.ndarray:
        """Calcula a saída para um vetor ou um lote de vetores."""
        
        entradas_array = np.asarray(entradas, dtype=float)

        if entradas_array.ndim not in (1, 2) or 0 in entradas_array.shape:
            raise ValueError("entradas deve ser um vetor ou lote não vazio.")

        if entradas_array.shape[-1] != self._pesos.shape[1]:
            raise ValueError(
                "A quantidade de entradas deve corresponder à quantidade de pesos."
            )

        if not np.isfinite(entradas_array).all():
            raise ValueError("Entradas devem conter números finitos.")

        resultado = np.dot(entradas_array, self._pesos.T) + self._bias

        #TODO Testar
        if ativacao is not None:
            return ativacao(resultado)

        return resultado