import numpy as np
from sklearn.datasets import load_digits

from mlp_app.core.camada import Camada
from mlp_app.core.rede_neural import RedeNeural
from mlp_app.settings import dados_exemplo as dados


class MLPController:

    def exibir_exemplos(self) -> None:
        digitos = load_digits()
        print(digitos.data[0])

        saida_neuronio = (
            np.dot(dados.PESOS_NEURONIO, dados.ENTRADAS) + dados.BIAS_NEURONIO
        )
        print(saida_neuronio)
        print(saida_neuronio)

        primeira_camada = Camada(dados.PESOS, dados.BIAS)
        saidas = primeira_camada.calcular(dados.ENTRADAS)
        print(saidas.tolist())
        print(saidas.tolist())
        print(type(saidas.tolist()))
        print(f"SHAPE INPUT: {np.shape(dados.ENTRADAS)}")
        print(saidas)
        print(type(saidas))
        print(primeira_camada.calcular(dados.ENTRADAS_LOTE))

        primeira_camada_lote = Camada(dados.PESOS_MAIORES, dados.BIAS)
        print(primeira_camada_lote.calcular(dados.ENTRADAS_LOTE_MAIOR))

        segunda_camada = Camada(
            dados.PESOS_SEGUNDA_CAMADA,
            dados.BIAS_SEGUNDA_CAMADA,
        )

        terceira_camada = Camada(
            dados.PESOS_TERCEIRA_CAMADA,
            dados.BIAS_TERCEIRA_CAMADA,
        )

        rede = RedeNeural([primeira_camada_lote, segunda_camada, terceira_camada])

        saida_final = rede.propagar(
            dados.ENTRADAS_LOTE_MAIOR,
            ativacoes=(None, None, lambda valores: np.maximum(0, valores)),
        )

        saida_segunda_camada = segunda_camada.calcular(
            primeira_camada_lote.calcular(dados.ENTRADAS_LOTE_MAIOR)
        )

        print("--------------------ANTES DA ATIVACAO---------------------------")
        print(terceira_camada.calcular(saida_segunda_camada))
        print("--------------------DEPOIS DA ATIVACAO---------------------------")
        print(saida_final)
