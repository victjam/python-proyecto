from scipy.optimize import linprog

# Resolver el problema primal
# Coeficientes de la función objetivo (minimización)
c_primal = [160, 120, 280]
# Restricciones del primal (se convierten >= en <= multiplicando por -1)
A_primal = [
    [-2, -1, -4],  # Restricción 1
    [-2, -2, -2]   # Restricción 2
]
b_primal = [-1, -1.5]  # Lado derecho de las restricciones

# Resolver el primal
primal_result = linprog(c=c_primal, A_ub=A_primal, b_ub=b_primal, bounds=(0, None))

# Mostrar resultados del primal
print("=== Solución Primal ===")
print(f"x1 = {primal_result.x[0]:.2f}")
print(f"x2 = {primal_result.x[1]:.2f}")
print(f"x3 = {primal_result.x[2]:.2f}")
print(f"Valor óptimo Z (mínimo): {primal_result.fun:.2f}")

# Resolver el problema dual
# Coeficientes de la función objetivo dual (maximización, por eso negativos para linprog)
c_dual = [-1, -1.5]
# Restricciones del dual (transpuesta de A_primal)
A_dual = [
    [2, 2],  # Restricción 1
    [1, 2],  # Restricción 2
    [4, 2]   # Restricción 3
]
b_dual = [160, 120, 280]  # Lado derecho de las restricciones del dual

# Resolver el dual
dual_result = linprog(c=c_dual, A_ub=A_dual, b_ub=b_dual, bounds=(0, None))

# Mostrar resultados del dual
print("\n=== Solución Dual ===")
print(f"y1 = {dual_result.x[0]:.2f}")
print(f"y2 = {dual_result.x[1]:.2f}")
print(f"Valor óptimo W (máximo): {-dual_result.fun:.2f}")  # Negamos el resultado porque linprog minimiza
