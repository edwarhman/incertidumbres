import pytest
import numpy as np
import pandas as pd
from sympy import symbols, sqrt, log
from src.incertidumbres import MedicionIndirecta, ProcesadorDatos

# Fixtures
@pytest.fixture
def sample_expression():
    x, y = symbols('x y')
    return x**2 + y, [x, y]

@pytest.fixture
def sample_data():
    return np.array([[4.0, 0.2], [2.0, 0.1]])  # valores, incertidumbres

class TestMedicionIndirecta:
    def test_evaluar_expresion(self, sample_expression):
        expr, vars = sample_expression
        result = MedicionIndirecta.evaluar_expresion(expr, vars, [2, 3])
        assert np.isclose(result, 7.0)  # 2² + 3 = 7
        
    def test_calcular_error_absoluto(self, sample_expression):
        expr, vars = sample_expression
        result = MedicionIndirecta.calcular_error_absoluto(
            expr, vars, [2, 3], [0.1, 0.2]
        )
        # Error absoluto = |∂f/∂x|Δx + |∂f/∂y|Δy
        # |2x|Δx + |1|Δy = |4|*0.1 + |1|*0.2 = 0.6
        assert np.isclose(result, 0.6, rtol=1e-5)
        
    def test_calcular_desviacion_estandar(self, sample_expression):
        expr, vars = sample_expression
        result = MedicionIndirecta.calcular_desviacion_estandar(
            expr, vars, [2, 3], [0.1, 0.2]
        )
        # σ = sqrt((∂f/∂x * Δx)² + (∂f/∂y * Δy)²)
        # sqrt((4*0.1)² + (1*0.2)²)
        expected = np.sqrt((4*0.1)**2 + (0.2)**2)
        assert np.isclose(result, expected, rtol=1e-1)
        
    def test_calcular_medicion(self, sample_expression):
        expr, vars = sample_expression
        datos = [[2, 3], [0.1, 0.2]]
        result = MedicionIndirecta.calcular_medicion(expr, vars, datos)
        assert len(result) == 2
        assert np.isclose(result[0], 7.0)  # valor
        # La incertidumbre debe ser la desviación estándar calculada antes
        expected_uncertainty = np.sqrt((4*0.1)**2 + (0.2)**2)
        assert np.isclose(result[1], expected_uncertainty, rtol=1e-1)
        
    def test_calcular_mediciones_lista(self, sample_expression):
        expr, vars = sample_expression
        datos_lista = [
            ([2, 3], [0.1, 0.2]),
            ([3, 1], [0.2, 0.1])
        ]
        results = MedicionIndirecta.calcular_mediciones_lista(expr, vars, datos_lista)
        assert results.shape == (2, 2)
        assert np.isclose(results[0, 0], 7.0)  # 2² + 3
        assert np.isclose(results[1, 0], 10.0)  # 3² + 1

class TestProcesadorDatos:
    def test_separar_valores_incertidumbres(self):
        datos = [1.0, 0.1, 2.0, 0.2, 3.0, 0.3]
        valores, incertidumbres = ProcesadorDatos.separar_valores_incertidumbres(datos)
        assert np.array_equal(valores, [1.0, 2.0, 3.0])
        assert np.array_equal(incertidumbres, [0.1, 0.2, 0.3])
        
    def test_procesar_dataframe(self, sample_expression):
        expr, vars = sample_expression
        # Crear DataFrame de prueba
        df = pd.DataFrame({
            'x': [2.0, 3.0],
            'Δx': [0.1, 0.2],
            'y': [3.0, 1.0],
            'Δy': [0.2, 0.1]
        })
        
        ProcesadorDatos.procesar_dataframe(
            df, expr, vars, 
            columnas=[0, 1, 2, 3], 
            nombre_resultado='resultado'
        )
        
        # Verificar que se agregaron las columnas correctamente
        assert 'resultado' in df.columns
        assert 'Δresultado' in df.columns
        assert np.isclose(df['resultado'].iloc[0], 7.0)  # 2² + 3
        assert np.isclose(df['resultado'].iloc[1], 10.0)  # 3² + 1

def test_ejemplo_practico():
    """Test de integración con un ejemplo práctico de física."""
    # Ejemplo: cálculo de la velocidad v = d/t
    d, t = symbols('d t')
    expr = d/t
    
    # Datos: distancia = 100 ± 1 m, tiempo = 10 ± 0.1 s
    datos = ([100, 10], [1, 0.1])
    
    resultado = MedicionIndirecta.calcular_medicion(expr, [d, t], datos)
    
    # Velocidad esperada: 10 m/s
    assert np.isclose(resultado[0], 10.0)
    
    # La incertidumbre se puede calcular con la fórmula de propagación
    # σv = v * sqrt((Δd/d)² + (Δt/t)²)
    expected_uncertainty = 10 * np.sqrt((1/100)**2 + (0.1/10)**2)
    assert np.isclose(resultado[1], expected_uncertainty, rtol=1e-2) 

def test_gravedad():
    """Test de integración con un ejemplo práctico de física."""
    # Ejemplo: cálculo de la gravedad
    y,t = symbols('y t')
    expr = 2*y/(t**2)
    
    # Datos: y = 10 m, t = 0.1 s
    datos = ([0.0532, 0.103], [0.02*10**-2, 0.001])
    
    resultado = MedicionIndirecta.calcular_medicion(expr, [y, t], datos)
    
    # Gravedad esperada: 20 m/s²
    assert np.isclose(resultado[0], 10.02922)
    
    # La incertidumbre se puede calcular con la fórmula de propagación
    # σg = g * sqrt((Δy/y)² + (Δt/t)²)
    expected_uncertainty = 19.8 * 10 ** (-2)
    assert np.isclose(resultado[1], expected_uncertainty, rtol=1e-1)

def test_ganancia_db():
    """Test de integración con un ejemplo práctico de física."""
    # Ejemplo: cálculo de la ganancia en dB
    A = symbols('A')
    expr = 20 * log(A, 10)
    
    # Datos: A = 2 V/V
    datos = ([2], [0.2236])
    
    resultado = MedicionIndirecta.calcular_medicion(expr, [A], datos)
    
    # Ganancia esperada: 6.020599913 dB
    assert np.isclose(resultado[0], 6.020599913)
    
    # # La incertidumbre se puede calcular con la fórmula de propagación
    # # σg = g * sqrt((Δy/y)² + (Δt/t)²)
    # expected_uncertainty = 0.1
    # assert np.isclose(resultado[1], expected_uncertainty, rtol=1e-2)