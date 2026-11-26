# Newton y Broyden para sistemas de ecuaciones no lineales en Python

[![CI](https://github.com/KevinAndrey96/python-newton-nonlinear-systems-solver/actions/workflows/ci.yml/badge.svg)](https://github.com/KevinAndrey96/python-newton-nonlinear-systems-solver/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-3776AB)
![Cobertura](https://img.shields.io/badge/cobertura-100%25-brightgreen)
![mypy](https://img.shields.io/badge/mypy-strict-2A6DB2)
[![Licencia: MIT](https://img.shields.io/badge/licencia-MIT-yellow.svg)](LICENSE)

Solucionador tipado y probado de sistemas $F(\mathbf{x}) = \mathbf{0}$ con $F: \mathbb{R}^n \to \mathbb{R}^n$, construido sobre NumPy: **método de Newton** con jacobiano analítico o por diferencias centrales, **búsqueda lineal de Armijo** para ampliar la región de convergencia y **método de Broyden** para ahorrar evaluaciones del jacobiano. Incluye los dos sistemas del trabajo original y una interfaz de línea de comandos que muestra cada iteración.

## Contenido

- [Los métodos](#los-métodos)
- [Resultados](#resultados)
- [Uso](#uso)
- [Arquitectura](#arquitectura)
- [Pruebas y calidad](#pruebas-y-calidad)
- [Historia del proyecto](#historia-del-proyecto)

## Los métodos

**Newton.** En cada iteración se linealiza $F$ alrededor de $\mathbf{x}_k$ y se resuelve el sistema lineal del paso, sin invertir nunca la matriz:

$$
J(\mathbf{x}_k)\,\mathbf{s}_k = -F(\mathbf{x}_k), \qquad \mathbf{x}_{k+1} = \mathbf{x}_k + t_k\,\mathbf{s}_k.
$$

Cerca de una raíz simple el error se eleva al cuadrado en cada paso. Si no se da el jacobiano, se aproxima por columnas con diferencias centrales y $h_j = \varepsilon^{1/3} \max(1, |x_j|)$, que equilibra el error de truncamiento $O(h^2)$ con el de redondeo.

**Búsqueda lineal.** El factor $t_k$ empieza en 1 y se divide a la mitad hasta cumplir la condición de Armijo sobre $\varphi(\mathbf{x}) = \tfrac{1}{2}\lVert F(\mathbf{x}) \rVert^2$:

$$
\lVert F(\mathbf{x}_k + t\,\mathbf{s}_k) \rVert^2 \le (1 - 2 \cdot 10^{-4}\, t)\,\lVert F(\mathbf{x}_k) \rVert^2.
$$

**Broyden.** Calcula el jacobiano una sola vez y luego lo corrige con actualizaciones de rango uno que respetan la ecuación secante $B_{k+1}\mathbf{s}_k = \mathbf{y}_k$, con $\mathbf{y}_k = F(\mathbf{x}_{k+1}) - F(\mathbf{x}_k)$:

$$
B_{k+1} = B_k + \frac{(\mathbf{y}_k - B_k \mathbf{s}_k)\,\mathbf{s}_k^\top}{\mathbf{s}_k^\top \mathbf{s}_k}.
$$

Cada iteración cuesta una sola evaluación de $F$ y la convergencia es superlineal en lugar de cuadrática.

La iteración se detiene cuando $\lVert F \rVert_\infty \le 10^{-12}$ (convergió), cuando el paso cae por debajo del redondeo (estancamiento, sin declarar éxito) o al agotar 50 iteraciones.

## Resultados

Iteraciones hasta $\lVert F \rVert_\infty \le 10^{-12}$ con los sistemas incluidos:

| Sistema | Punto inicial | Newton | Newton sin búsqueda lineal | Broyden | Broyden sin búsqueda lineal |
| :--- | :--- | --: | --: | --: | --: |
| `tres-incognitas` | $(1.8, 3.2, 4.2)$ | 8 | 9 | 14 | 24 |
| `parabola-seno` | $(1, 3)$ | 3 | 3 | 5 | 5 |
| `circulo-recta` | $(1, 2)$ | 5 | 5 | 6 | 6 |
| `rosenbrock` | $(-1.2, 1)$ | 10 | 2 | 22 | 3 |
| `arcotangente` | $(2, -3)$ | 7 | diverge | 11 | no converge |

Los dos primeros son los del trabajo original. El sistema de tres incógnitas

$$
x_1^2 + x_2 x_3 = 16, \qquad x_1 x_2 + x_3 = 10, \qquad x_1 + x_2^2 + 24 x_3 = 15
$$

tiene la raíz $(3.936375765277,\ 2.488867635853,\ 0.202881755247)$, y el segundo, $(1.099743712038,\ 2.899436232167)$. La tabla muestra también el compromiso de la búsqueda lineal: rescata el arranque lejano de `arcotangente`, donde Newton puro se aleja hasta que el jacobiano se anula, pero en `rosenbrock` rechaza pasos completos que aumentan $\lVert F \rVert$ momentáneamente y que habrían llevado a la raíz en dos iteraciones. Por eso se puede desactivar.

## Uso

Requiere Python 3.12 o superior y [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run newton-systems list
uv run newton-systems solve parabola-seno
#   k  x                      ||F||inf        paso       t
#   0  1 3                   2.871e+00   0.000e+00       1
#   1  1.09988429009 ...     1.358e-02   1.102e-01       1
#   2  1.09974371071 ...     1.976e-08   9.668e-03       1
#   3  1.09974371204 ...     3.553e-15   2.269e-08       1
# convergió en 3 iteraciones
uv run newton-systems solve rosenbrock --no-line-search
uv run newton-systems solve tres-incognitas -m broyden --numeric-jacobian --x0 "4 2 0"
```

El comando termina con código 0 si converge, 1 si no y 2 ante un jacobiano singular o un punto inicial inválido.

Como librería, con cualquier función que devuelva un vector:

```python
import numpy as np
from newton_systems import Options, broyden, newton

def F(x):
    return [x[0] ** 2 + x[1] ** 2 - 4, np.exp(x[0]) + x[1] - 1]

resultado = newton(F, [1.0, -1.0])                      # jacobiano numérico
resultado.x, resultado.converged, resultado.iterations
broyden(F, [1.0, -1.0], options=Options(tolerance=1e-10))
for paso in resultado.history:
    print(paso.iteration, paso.x, paso.residual_norm, paso.damping)
```

## Arquitectura

```text
src/newton_systems/
  solver.py     Newton, Broyden, jacobiano numérico, búsqueda lineal e historial
  examples.py   sistemas del trabajo original y casos clásicos con su jacobiano
  cli.py        comandos list y solve con tabla de iteraciones
```

Los dos métodos comparten el mismo bucle y solo difieren en cómo obtienen la dirección y si actualizan su aproximación del jacobiano. Los errores se reportan con `SingularJacobianError` y `DimensionError`.

## Pruebas y calidad

```bash
uv run pytest --cov
uv run ruff check src tests && uv run ruff format --check src tests
uv run mypy
```

33 pruebas con cobertura del 100 % de líneas y ramas. Verifican las raíces de los sistemas originales, la convergencia cuadrática midiendo $e_{k+1}/e_k^2 \to 1/(2\sqrt{2})$ en el sistema círculo-recta, que el jacobiano numérico coincide con el analítico, que la búsqueda lineal rescata el arranque divergente y, con Hypothesis, que cualquier sistema afín bien condicionado se resuelve en una iteración. El pipeline de GitHub Actions ejecuta Ruff y mypy en modo estricto, las pruebas en Python 3.12, 3.13 y 3.14, una prueba de humo del comando instalado y la construcción del paquete con `twine check`.

## Historia del proyecto

El proyecto nació en julio de 2020 como un ejercicio de métodos numéricos: un script que desarrollaba a mano tres iteraciones de Newton para cada sistema, redondeando a cuatro decimales en cada paso e invirtiendo el jacobiano con `numpy.matrix`. Con tres iteraciones fijas el sistema de tres incógnitas seguía lejos de la raíz (el residuo superaba 50 en la tercera). En 2026 se reescribió como un paquete con iteración hasta convergencia, solución del sistema lineal en lugar de la inversa, jacobiano numérico, búsqueda lineal, método de Broyden y pruebas automatizadas; además se eliminaron del historial el entorno virtual y los archivos del IDE que se habían versionado. El código original se conserva en el tag [`v0.1-python-coursework-2020`](https://github.com/KevinAndrey96/python-newton-nonlinear-systems-solver/tree/v0.1-python-coursework-2020).

## Autor

**Kevin Andrey Herrera Silva** · [GitHub](https://github.com/KevinAndrey96)

## Licencia

Distribuido bajo la [licencia MIT](LICENSE).
