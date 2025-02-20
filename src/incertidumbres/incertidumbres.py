import numpy as np
from sympy import diff, Abs, N
import math

class MedicionIndirecta:
    """Clase para realizar cálculos de mediciones indirectas y propagación de errores."""
    
    @staticmethod
    def evaluar_expresion(expresion, variables, valores):
        """Evalúa una expresión simbólica con los valores dados."""
        return N(expresion.subs(dict(zip(variables, valores))))

    @staticmethod
    def calcular_error_absoluto(expresion, variables, valores, incertidumbres):
        """Calcula el error absoluto usando el método de derivadas parciales."""
        derivadas = [Abs(diff(expresion, var)) * dx 
                    for var, dx in zip(variables, incertidumbres)]
        
        subs_dict = dict(zip(variables, valores))
        return sum(N(deriv.subs(subs_dict)) for deriv in derivadas)

    @staticmethod
    def calcular_desviacion_estandar(expresion, variables, valores, incertidumbres):
        """Calcula la desviación estándar usando propagación de errores."""
        derivadas = [Abs(diff(expresion, var)) * dx 
                    for var, dx in zip(variables, incertidumbres)]
        
        subs_dict = dict(zip(variables, valores))
        coefs = np.array([N(deriv.subs(subs_dict)) for deriv in derivadas])
        suma = np.sum(coefs**2)
        return suma ** (1/2)

    @classmethod
    def calcular_medicion(cls, expresion, variables, datos):
        """Calcula el valor y la incertidumbre de una medición indirecta."""
        valores, incertidumbres = datos
        valor = cls.evaluar_expresion(expresion, variables, valores)
        incertidumbre = cls.calcular_desviacion_estandar(expresion, variables, valores, incertidumbres)
        return np.array([valor, incertidumbre])

    @classmethod
    def calcular_mediciones_lista(cls, expresion, variables, datos_lista):
        """Calcula mediciones indirectas para una lista de datos."""
        return np.array([cls.calcular_medicion(expresion, variables, datos) 
                        for datos in datos_lista])

class ProcesadorDatos:
    """Clase para procesar y transformar datos de mediciones."""
    
    @staticmethod
    def separar_valores_incertidumbres(datos):
        """Separa una lista alternada de valores e incertidumbres."""
        return np.array([datos[::2], datos[1::2]])

    @staticmethod
    def procesar_dataframe(df, expresion, variables, columnas, nombre_resultado):
        """Agrega columnas de resultado y su incertidumbre a un DataFrame."""
        datos = df.to_numpy()[:, columnas]
        datos_separados = np.array([ProcesadorDatos.separar_valores_incertidumbres(fila) 
                                  for fila in datos])
        
        resultados = MedicionIndirecta.calcular_mediciones_lista(
            expresion, variables, datos_separados)
        
        df[nombre_resultado] = resultados[:, 0]
        df[f'Δ{nombre_resultado}'] = resultados[:, 1]