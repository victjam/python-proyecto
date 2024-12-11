from scipy.optimize import linprog

# Definición de los parámetros del problema
# Precios de los arreglos
P1, P2 = 50, 40  # Ingresos por cada tipo de arreglo floral

# Disponibilidad de flores
b1, b2, b3 = 500, 300, 200  # Rosas, Tulipanes, Hibiscos

# Requerimientos de flores por arreglo
a11, a12 = 5, 4  # Rosas
a21, a22 = 3, 2  # Tulipanes
a31, a32 = 1, 3  # Hibiscos

# Resolver el problema primal
c_primal = [-P1, -P2]  # Negativos porque linprog minimiza por defecto
A_primal = [
    [a11, a12],
    [a21, a22],
    [a31, a32]
]
b_primal = [b1, b2, b3]

# Resolver el problema primal
res_primal = linprog(c=c_primal, A_ub=A_primal, b_ub=b_primal, bounds=(0, None))

# Resolver el problema dual
c_dual = [b1, b2, b3]  # Coeficientes del dual (ingresos por unidad adicional)
A_dual = [
    [a11, a21, a31],  # Transpuesta de la matriz primal (rosas)
    [a12, a22, a32]   # Transpuesta de la matriz primal (tulipanes)
]
b_dual = [P1, P2]  # Precios de los arreglos

# Convertir A_dual a la forma correcta
A_dual = list(map(list, zip(*A_dual)))  # Transponer A_dual para que coincida con c_dual

# Resolver el problema dual
res_dual = linprog(c=c_dual, A_ub=-1 * A_dual, b_ub=-1 * b_dual, bounds=(0, None))

# Mostrar resultados del primal
print("=== Solución Primal ===")
print(f"x1 (arreglos del tipo 1): {res_primal.x[0]:.2f}")
print(f"x2 (arreglos del tipo 2): {res_primal.x[1]:.2f}")
print(f"Valor óptimo Z (ingresos máximos): {-res_primal.fun:.2f}")

# Mostrar resultados del dual
print("\n=== Solución Dual ===")
print(f"y1 (precio sombra de las rosas): {res_dual.x[0]:.2f}")
print(f"y2 (precio sombra de los tulipanes): {res_dual.x[1]:.2f}")
print(f"y3 (precio sombra de los hibiscos): {res_dual.x[2]:.2f}")
print(f"Valor óptimo W (costo mínimo): {res_dual.fun:.2f}")
