# Método de Newton-Raphson para Sistemas de Ecuaciones No Lineales en Python

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Supported-013243.svg)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Implementación del método multidimensional de **Newton-Raphson** para la resolución numérica de sistemas acoplados de ecuaciones algebraicas no lineales en Python utilizando álgebra matricial con **NumPy**. El proyecto ilustra el cálculo explícito de la matriz Jacobiana, la inversión matricial y la convergencia iterativa paso a paso.

---

## 📐 Fundamento Teórico y Matemático

El método multidimensional de Newton-Raphson generaliza la búsqueda de raíces unidimensional al espacio euclidiano $\mathbb{R}^n$. Dado un sistema de $n$ ecuaciones no lineales con $n$ incógnitas:

$$\mathbf{F}(\mathbf{x}) = \mathbf{0}, \quad \text{donde } \mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}, \quad \mathbf{F}(\mathbf{x}) = \begin{bmatrix} f_1(\mathbf{x}) \\ f_2(\mathbf{x}) \\ \vdots \\ f_n(\mathbf{x}) \end{bmatrix}$$

### 1. Linealización Mediante Serie de Taylor

Aproximando $\mathbf{F}(\mathbf{x})$ alrededor del vector de estimación actual $\mathbf{x}^{(k)}$:

$$\mathbf{F}(\mathbf{x}^{(k)} + \Delta \mathbf{x}) \approx \mathbf{F}(\mathbf{x}^{(k)}) + \mathbf{J}(\mathbf{x}^{(k)}) \Delta \mathbf{x} = \mathbf{0}$$

Donde $\mathbf{J}(\mathbf{x})$ es la **matriz Jacobiana** de derivadas parciales:

$$\mathbf{J}(\mathbf{x}) = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_n}{\partial x_1} & \frac{\partial f_n}{\partial x_2} & \cdots & \frac{\partial f_n}{\partial x_n}
\end{bmatrix}$$

### 2. Esquema de Actualización Iterativa

En cada iteración $k$, se resuelve el sistema lineal de corrección para $\Delta \mathbf{x}^{(k)}$ y se actualiza el vector solución:

$$\mathbf{J}(\mathbf{x}^{(k)}) \Delta \mathbf{x}^{(k)} = -\mathbf{F}(\mathbf{x}^{(k)}) \implies \Delta \mathbf{x}^{(k)} = -\left[ \mathbf{J}(\mathbf{x}^{(k)}) \right]^{-1} \mathbf{F}(\mathbf{x}^{(k)})$$
$$\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} + \Delta \mathbf{x}^{(k)}$$

---

## 🔬 Sistemas Resueltos en el Repositorio

El script `main.py` desarrolla dos casos de estudio con deducción analítica de la matriz Jacobiana:

### Caso 1: Sistema No Lineal $3 \times 3$

$$\begin{cases}
f_1(x_1, x_2, x_3) = x_1^2 + x_2 x_3 - 16 = 0 \\
f_2(x_1, x_2, x_3) = x_1 x_2 + x_3 - 10 = 0 \\
f_3(x_1, x_2, x_3) = x_1 + x_2^2 + 24 x_3 - 15 = 0
\end{cases}$$

Matriz Jacobiana analítica asociada:

$$\mathbf{J}(x_1, x_2, x_3) = \begin{bmatrix}
2 x_1 & x_3 & x_2 \\
x_2 & x_1 & 1 \\
1 & 2 x_2 & 24
\end{bmatrix}$$

### Caso 2: Sistema No Lineal Trigonométrico $2 \times 2$

$$\begin{cases}
g_1(x_1, x_2) = x_1^2 - x_2 + 1.69 = 0 \\
g_2(x_1, x_2) = 20 x_1 \sin(x_1) - x_2 - 16.7 = 0
\end{cases}$$

Matriz Jacobiana asociada:

$$\mathbf{J}(x_1, x_2) = \begin{bmatrix}
2 x_1 & -1 \\
20 \sin(x_1) + 20 x_1 \cos(x_1) & -1
\end{bmatrix}$$

---

## 📂 Estructura del Proyecto

```text
newton-sistema-ecuaciones/
└── main.py       # Script principal de resolución iterativa con NumPy
```

---

## 🚀 Requisitos y Ejecución

### Prerrequisitos

- **Python** (versión 3.8 o superior).
- Biblioteca **NumPy**.

### Instalación de Dependencias

```bash
pip install numpy
```

### Ejecución del Script

```bash
python main.py
```

### Salida de Ejemplo (Consola)

```text
x1:  1.8 		x2: 3.2 		x3: 4.2
fx
[[ 9.68]
 [ 0.  ]
 [-0.96]]
fpx
[[ 3.6  4.2  3.2]
 [ 3.2  1.8  1. ]
 [ 1.   6.4 24. ]]
deltx
[[-1.8727]
 [ 0.8845]
 [-0.1708]]
Segunda Iteración
x1:  -0.0727 		x2: 4.0845 		x3: 4.0292
...
```

---

## 📄 Licencia

Distribuido bajo la Licencia MIT. Consulta el archivo `LICENSE` para más información.
