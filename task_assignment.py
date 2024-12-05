import numpy as np

class TaskAssignmentOptimizer:
    """
    Clase para optimizar la asignación de tareas entre programadores.
    """

    def __init__(self):
        # Matriz de costos o tiempos
        self.matrix = None
        # Número de programadores y tareas
        self.num_programmers = 0
        self.num_tasks = 0
        # Criterio de optimización ('1' para Tiempo, '2' para Costo)
        self.criterio = '1'  # Valor por defecto

    def get_user_matrix(self):
        """
        Solicita al usuario ingresar la matriz de costos o tiempos.
        """
        try:
            self.num_programmers = int(input("Ingrese el número de programadores: "))
            self.num_tasks = int(input("Ingrese el número de tareas: "))

            if self.num_programmers <= 0 or self.num_tasks <= 0:
                print("El número de programadores y tareas debe ser mayor a cero.")
                return False

            self.matrix = []

            print("\nIngrese los valores de la matriz:")
            for i in range(self.num_programmers):
                row = []
                print(f"\nProgramador {i+1}:")
                for j in range(self.num_tasks):
                    while True:
                        try:
                            value = float(input(f"Costo/Tiempo para la tarea {j+1}: "))
                            if value < 0:
                                print("El valor no puede ser negativo. Intente nuevamente.")
                            else:
                                row.append(value)
                                break
                        except ValueError:
                            print("Entrada inválida. Ingrese un número.")
                self.matrix.append(row)
            
            # Convertimos la matriz a un numpy array
            self.matrix = np.array(self.matrix)
            return True
        except ValueError:
            print("Entrada inválida. Por favor, ingrese números enteros.")
            return False

    def validate_matrix(self):
        """
        Valida que la matriz esté correctamente formada y que los datos sean consistentes.
        """
        if not isinstance(self.matrix, np.ndarray) or self.matrix.size == 0:
            print("Error: La matriz no puede estar vacía.")
            return False

        if self.matrix.shape[0] != self.num_programmers or self.matrix.shape[1] != self.num_tasks:
            print("Error: La matriz no tiene las dimensiones correctas.")
            return False

        if np.any(self.matrix < 0):
            print("Error: Los valores no pueden ser negativos.")
            return False

        return True

    def balance_matrix(self):
        """
        Balancea la matriz agregando filas o columnas ficticias si es necesario.
        """
        rows, cols = self.matrix.shape
        if rows == cols:
            return
        max_dim = max(rows, cols)
        # Usamos un valor grande para las posiciones ficticias
        fill_value = np.max(self.matrix) * 10 + 1
        balanced_matrix = np.full((max_dim, max_dim), fill_value=fill_value)

        # Copiamos los valores originales
        balanced_matrix[:rows, :cols] = self.matrix
        self.matrix = balanced_matrix
        self.num_programmers = self.num_tasks = max_dim

    def subtract_row_minima(self):
        """
        Restamos el mínimo de cada fila a todos los elementos de esa fila.
        """
        row_minima = np.min(self.matrix, axis=1)
        self.matrix -= row_minima[:, np.newaxis]

    def subtract_column_minima(self):
        """
        Restamos el mínimo de cada columna a todos los elementos de esa columna.
        """
        col_minima = np.min(self.matrix, axis=0)
        self.matrix -= col_minima

    def find_optimal_assignment_manual(self):
        """
        Encuentra una asignación óptima utilizando el Algoritmo Húngaro manualmente.
        """
        n = self.num_programmers
        # Paso 1: Restar los mínimos de filas y columnas
        self.subtract_row_minima()
        self.subtract_column_minima()

        # Paso 2: Inicialización
        assignment = -np.ones(n, dtype=int)
        lines = 0

        while True:
            # Paso 3: Marcar ceros primos y estrellas
            stars = np.zeros_like(self.matrix, dtype=bool)
            primes = np.zeros_like(self.matrix, dtype=bool)
            covered_rows = np.zeros(n, dtype=bool)
            covered_cols = np.zeros(n, dtype=bool)

            # Estrella los ceros únicos en su fila
            for i in range(n):
                for j in range(n):
                    if self.matrix[i, j] == 0 and not covered_cols[j] and not covered_rows[i]:
                        stars[i, j] = True
                        covered_cols[j] = True
                        covered_rows[i] = True

            # Reiniciar las coberturas
            covered_rows[:] = False
            covered_cols[:] = False

            # Paso 4: Cubrir columnas con estrellas
            for j in range(n):
                if stars[:, j].any():
                    covered_cols[j] = True

            # Paso 5: Bucle principal
            while True:
                if covered_cols.sum() == n:
                    # Encontramos una solución completa
                    break

                # Encontrar un cero no cubierto y marcarlo como primo
                zero_found = False
                zeros = np.where((self.matrix == 0) & (~covered_rows[:, np.newaxis]) & (~covered_cols))
                if zeros[0].size == 0:
                    zero_found = False
                else:
                    zero_found = True
                    row = zeros[0][0]
                    col = zeros[1][0]
                    primes[row, col] = True

                    # Si hay una estrella en esa fila
                    star_col = np.where(stars[row])[0]
                    if star_col.size > 0:
                        covered_rows[row] = True
                        covered_cols[star_col[0]] = False
                    else:
                        # Paso 6: Construir el camino alternante
                        path = [(row, col)]
                        while True:
                            star_row = np.where(stars[:, path[-1][1]])[0]
                            if star_row.size == 0:
                                break
                            else:
                                path.append((star_row[0], path[-1][1]))
                                prime_col = np.where(primes[path[-1][0]])[0]
                                path.append((path[-1][0], prime_col[0]))
                        # Alternar estrellas y primos en el camino
                        for r, c in path:
                            stars[r, c] = not stars[r, c]
                        # Limpiar las marcas
                        primes[:, :] = False
                        covered_rows[:] = False
                        covered_cols[:] = False
                        # Cubrir columnas con estrellas
                        for j in range(n):
                            if stars[:, j].any():
                                covered_cols[j] = True
                        break
                if not zero_found:
                    # Paso 7: Modificar la matriz para crear ceros adicionales
                    min_uncovered = np.min(self.matrix[~covered_rows][:, ~covered_cols])
                    self.matrix[covered_rows] += min_uncovered
                    self.matrix[:, ~covered_cols] -= min_uncovered
                else:
                    continue
                if covered_cols.sum() == n:
                    # Encontramos una solución completa
                    break

            # Salir del bucle principal si tenemos una solución completa
            if covered_cols.sum() == n:
                break

        # Construir la asignación a partir de las estrellas
        solution = []
        total_cost = 0
        for i in range(n):
            j = np.where(stars[i])[0]
            if j.size > 0:
                if i < self.original_matrix.shape[0] and j[0] < self.original_matrix.shape[1]:
                    total_cost += self.original_matrix[i, j[0]]
                    solution.append((i, j[0]))

        return solution, total_cost

    def find_optimal_assignment_library(self):
        """
        Encuentra una asignación óptima utilizando la librería SciPy.
        """
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            print("La librería 'scipy' no está instalada. Por favor, instálala usando 'pip install scipy'.")
            return None

        # Asegurarse de que la matriz esté balanceada
        if self.num_programmers != self.num_tasks:
            self.balance_matrix()

        # Utilizar la función linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(self.matrix)
        total_cost = 0
        solution = []
        for i, j in zip(row_ind, col_ind):
            if i < self.original_matrix.shape[0] and j < self.original_matrix.shape[1]:
                total_cost += self.original_matrix[i, j]
                solution.append((i, j))

        return solution, total_cost

    def optimize_assignment(self, method='manual'):
        """
        Optimiza la asignación de tareas según el método seleccionado.
        """
        if not self.validate_matrix():
            return

        self.original_matrix = np.copy(self.matrix)

        # Modificar la matriz según el criterio seleccionado
        if self.criterio == '1':
            print("\nOptimizando Tiempo (Minimizando)...")
            # No se requiere ninguna modificación adicional
        elif self.criterio == '2':
            print("\nOptimizando Costo (Maximizando)...")
            # Convertir el problema de maximización en minimización
            max_value = np.max(self.matrix)
            self.matrix = max_value - self.matrix
        else:
            print("\nSelección inválida. Se asumirá minimización del Tiempo.")
            # Se asume minimización

        if method == 'manual':
            result = self.find_optimal_assignment_manual()
        elif method == 'munkres':
            result = self.find_optimal_assignment_library()
        else:
            print("Método no implementado.")
            return

        if result is None:
            return

        solution, total_cost = result

        # Si invertimos la matriz para maximizar, recalculamos el costo total correctamente
        if self.criterio == '2':
            total_cost = 0
            for programmer, task in solution:
                if programmer < self.original_matrix.shape[0] and task < self.original_matrix.shape[1]:
                    total_cost += self.original_matrix[programmer, task]

        print("\nAsignación óptima:")
        for programmer, task in solution:
            if programmer < self.original_matrix.shape[0] and task < self.original_matrix.shape[1]:
                print(f"Tarea {task+1} asignada al Programador {programmer+1}")
        print(f"\nCosto total: {total_cost}")

def main():
    optimizer = TaskAssignmentOptimizer()

    if not optimizer.get_user_matrix():
        return

    print("\nSeleccione el criterio de optimización:")
    print("1. Tiempo (Minimizar)")
    print("2. Costo (Maximizar)")
    criterio = input("Ingrese su elección (1/2): ")
    optimizer.criterio = criterio  # Guardamos la elección en el objeto

    print("\nSeleccione el método de resolución:")
    print("1. Usar la librería Munkres")
    print("2. Resolver sin librerías")
    method_choice = input("Ingrese su elección (1/2): ")

    method = 'munkres' if method_choice == '1' else 'manual'
    optimizer.optimize_assignment(method)

if __name__ == "__main__":
    main()
