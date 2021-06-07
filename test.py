
# Knapsack problem with additional constraints.

import dimod
from dimod.reference.samplers import ExactSolver

from dwave.system import LeapHybridSampler, LeapHybridDQMSampler

from math import log2, floor
import neal

import sys


value = [5, 5, 7, 10, 6, 6, 5, 6, 4, 3, 7, 9, 7, 5, 7, 8, 5, 8, 10,
         6, 4, 8, 8, 8, 9, 2, 10, 1, 5, 4, 6, 10, 10, 4, 5, 4, 7, 10,
         6, 7, 7, 4, 1, 3, 6, 2, 2, 4, 6, 2, 4, 6, 1, 2, 2, 1, 4, 2, 7, 
         5, 6, 10, 4, 9, 9, 10, 1, 9, 4, 2, 5, 3, 5, 4, 6, 3, 7, 9, 2, 7, 
         1, 10, 1, 5, 6, 5, 4, 8, 9, 7, 2, 10, 2, 7, 3, 10, 4, 3, 5, 5]

weight = [3, 4, 7, 9, 12, 3, 4, 7, 9, 12, 3, 4, 7, 9, 12, 3, 4, 7, 9, 12, 
          3, 4, 7, 9, 12, 3, 4, 7, 9, 12, 3, 4, 7, 9, 12, 3, 4, 7, 9, 12, 
          3, 4, 7, 9, 12, 3, 4, 7, 9, 12, 3, 4, 7, 9, 12, 3, 4, 7, 9, 12, 
          3, 4, 7, 9, 12, 3, 4, 7, 9, 12, 3, 4, 7, 9, 12, 3, 4, 7, 9, 12, 
          3, 4, 7, 9, 12, 3, 4, 7, 9, 12, 3, 4, 7, 9, 12, 3, 4, 7, 9, 12]

w_limit = 350
penalty = 100

constraint_owner = [(1,0), (2,0), (2,1), (3,0), (3,1), (3,2), (4,0), (4,1), 
                    (4,2), (4,3), (6,5), (7,5), (7,6), (8,5), (8,6), (8,7),
					(9,5), (9,6), (9,7), (9,8), (11,10), (12,10), (12,11), (13,10),
					(13,11), (13,12), (14,10), (14,11), (14,12), (14,13), (16,15), (17,15), (17,16),
					(18,15), (18,16), (18,17), (19,15), (19,16), (19,17), (19,18), (21,20), (22,20),
					(22,21), (23,20), (23,21), (23,22), (24,20), (24,21), (24,22), (24,23), (26,25),
					(27,25), (27,26), (28,25), (28,26), (28,27), (29,25), (29,26), (29,27), (29,28),
					(31,30), (32,30), (32,31), (33,30), (33,31), (33,32), (34,30), (34,31), (34,32),
					(34,33), (36,35), (37,35), (37,36), (38,35), (38,36), (38,37), (39,35), (39,36),
					(39,37), (39,38), (41,40), (42,40), (42,41), (43,40), (43,41), (43,42), (44,40),
					(44,41), (44,42), (44,43), (46,45), (47,45), (47,46), (48,45), (48,46), (48,47),
					(49,45), (49,46), (49,47), (49,48), (51,50), (52,50), (52,51), (53,50), (53,51),
					(53,52), (54,50), (54,51), (54,52), (54,53), (56,55), (57,55), (57,56), (58,55),
					(58,56), (58,57), (59,55), (59,56), (59,57), (59,58), (61,60), (62,60), (62,61),
					(63,60), (63,61), (63,62), (64,60), (64,61), (64,62), (64,63), (66,65), (67,65),
					(67,66), (68,65), (68,66), (68,67), (70,69), (71,69), (71,70), (72,69), (72,70),
					(72,71), (73,69), (73,70), (73,71), (73,72), (74,69), (74,70), (74,71), (74,72),
					(74,73), (76,75), (77,75), (77,76), (78,75), (78,76), (78,77), (79,75), (79,76),
					(79,77), (79,78), (81,80), (82,80), (82,81), (83,80), (83,81), (83,82), (84,80),
					(84,81), (84,82), (84,83), (86,85), (87,85), (87,86), (88,85), (88,86), (88,87),
					(89,85), (89,86), (89,87), (89,88), (91,90), (92,90), (92,91), (93,90), (93,91),
					(93,92), (94,90), (94,91), (94,92), (94,93), (96,95), (97,95), (97,96), (98,95),
					(98,96), (98,97), (99,95), (99,96), (99,97), (99,98)]

constraint_funct = [(10,7), (14,10), (17,3), (17,13), (27,19), (28,6), (28,25),
					(31,12), (34,14), (34,12), (36,24), (36,33), (38,32), (38,29), 
					(40,7), (40,38), (41,27), (43,25), (43,27), (46,24), (47,19), 
					(47,18), (47,37), (48,18), (48,47), (49,25), (50,47), (50,37), 
					(51,6), (51,36), (51,17), (52,14), (55,16), (56,48), (56,47), 
					(56,54), (56,18), (57,48), (57,18), (58,50), (58,47), (59,24), 
					(59,38), (59,50), (59,0), (60,34), (60,59), (60,0), (62,34),
					(63,8), (63,51), (63,37), (64,51), (64,63), (66,30), (67,63),
					(67,36), (69,27), (69,41), (70,45), (70,49), (71,21), (72,20),
					(73,24), (74,49), (74,51), (74,3), (75,13), (76,72), (76,37),
					(77,13), (78,34), (79,61), (81,45), (82,8), (82,44), (82,65),
					(85,25), (85,47), (85,37), (85,59), (86,17), (86,13), (88,68),
					(88,13), (89,43), (89,6), (89,25), (89,13), (89,45), (89,82),
					(89,44), (89,56), (90,41), (92,0), (92,82), (92,8), (92,58),
					(93,30), (93,89), (93,45), (94,11), (95,18), (96,12), (96,74),
					(98,75), (99,14), (99,72), (99,88)]


def build_qubo_bqm(value, weight,
                   constraint_owner,
                   constraint_funct,
                   w_limit,
                   penalty=10):
    # Initialize BQM - use large-capacity BQM so that the problem can be
    # scaled by the user.
    # * - Ahora mismo las variables pueden ser tipo Vartype.SPIN o Vartype.BINARY - *
    # Investigar si se puede alimentar al modelo con un ¿diccionario? con una serie de elementos
    # discretos entre los que la solución del problema tenga que elegir
    bqm = dimod.AdjVectorBQM(dimod.Vartype.BINARY)

    # Lagrangian multiplier
    # First guess as suggested in Lucas's paper
    # * - Transformamos un problema de k variables con restricciones en uno de n+k variables sin restricciones
    # * - Cada una de esas nuevas variables escalares, una por restricción, serán los multiplicadores de Langrange 
    lagrange = max(value)
    #print("Maximun cost is {}".format(lagrange))

    # Lucas's algorithm introduces additional slack variables to
    # handle the inequality. M+1 binary slack variables are needed to
    # represent the sum using a set of powers of 2.
    # * - Para una capacidad de 1000 unidades de carga, necesitaremos 9 + 1 variables
    M = floor(log2(w_limit))
    num_slack_variables = M + 1

    # Slack variable list for Lucas's algorithm. The last variable has
    # a special value because it terminates the sequence.
    # * - Definimos las variables de holgura
    # * - sum(y) = weight_capacity
    y = [2**n for n in range(M)]
    y.append(w_limit + 1 - 2**M)

    # Hamiltonian xi-xi terms
    # - * Lagrange * (peso^2 - valor)
    for index, (v, w) in enumerate(zip(value, weight)):
        bqm.set_linear('x' + str(index), lagrange * (w**2) - v)

    # Hamiltonian xi-xj terms
    # Evitan que se seleccionen ítems de más 
    for i in range(len(weight)):
        for j in range(i + 1, len(weight)):
            key = ('x' + str(i), 'x' + str(j))
            #print(key)
            weight1 = weight[i]
            weight2 = weight[j]
            bqm.quadratic[key] = 2 * lagrange * weight1 * weight2


    # Constraints associated to items of the same owner:
    for constr in constraint_owner:
        key = ('x' + str(constr[0]), 'x' + str(constr[1]))
        weight1 = weight[constr[0]]
        weight2 = weight[constr[1]]
        bqm.quadratic[key] = penalty * lagrange * weight1 * weight2      

    # Constraints associated to items with the same function
    for constr in constraint_funct:
        key = ('x' + str(constr[0]), 'x' + str(constr[1]))
        weight1 = weight[constr[0]]
        weight2 = weight[constr[1]]
        bqm.quadratic[key] = penalty * lagrange * weight1 * weight2   


    # Hamiltonian y-y terms
    for k in range(num_slack_variables):
        bqm.set_linear('y' + str(k), lagrange * (y[k]**2))

    # Hamiltonian yi-yj terms
    for i in range(num_slack_variables):
        for j in range(i + 1, num_slack_variables):
            key = ('y' + str(i), 'y' + str(j))
            bqm.quadratic[key] = 2 * lagrange * y[i] * y[j]

    # Hamiltonian x-y terms
    for index, w in enumerate(weight):
        for j in range(num_slack_variables):
            key = ('x' + str(index), 'y' + str(j))
            bqm.quadratic[key] = -2 * lagrange * w * y[j]

    return bqm



def solve_qubo(value, weight,
               constraint_owner,
               constraint_funct,
               w_limit, sampler=None):
    
    """Construct BQM and solve the knapsack problem
    
    Args:
        values (array-like):
            Array of values associated with the items
        weights (array-like):
            Array of weights associated with the items
        constraint_owner (array-like)
            Array that informs if two items belong to the same owner
        constraint_colour (array-like)
            Array that informs if two items are the same colour
        w_limit (int):
            Maximum allowable weight
        sampler (BQM sampler instance or None):
            A BQM sampler instance or None, in which case
            LeapHybridSampler is used by default
    
    Returns:
        Tuple:
            List of indices of selected items
            Solution energy
    """
    bqm = build_qubo_bqm(value, 
    	                 weight,
                         constraint_owner,
                         constraint_funct,
                         w_limit)

    if sampler == "neal":
        sampler = neal.SimulatedAnnealingSampler()
        sampleset = sampler.sample(bqm)
    elif sampler == "exact":
        if len(df_actions)<=20:
            sampler = ExactSolver()
            sampleset = sampler.sample(bqm)
        else:
            return None, None
    else:
        sampler = LeapHybridSampler()
        sampleset = sampler.sample(bqm)

    
    sample = sampleset.first.sample
    energy = sampleset.first.energy

    # Build solution from returned binary variables:
    selected_item_indices = []
    for varname, value in sample.items():
        # For each "x" variable, check whether its value is set, which
        # indicates that the corresponding item is included in the
        # knapsack
        if value and varname.startswith('x'):
            # The index into the weight array is retrieved from the
            # variable name
            selected_item_indices.append(int(varname[1:]))

    return sorted(selected_item_indices), energy


best_energy = 0
sampler_type ="neal"
num_iter = 100
if sampler_type =="exact":
    num_iter = 1

for i in range(num_iter):
    #print(i)
    selected_item_indices, energy = solve_qubo(value, 
    	                                       weight,
                                               constraint_owner,
                                               constraint_funct,
                                               w_limit, sampler_type)

    if selected_item_indices is not None:
        if energy < best_energy:
            print(i)
            best_energy = energy
            final_items = [ele for ele in selected_item_indices]
            selected_costs = [weight[i] for i in final_items]   
            selected_returns = [value[i] for i in final_items]  
            print("Found solution at energy {}".format(energy))
            print("Selected item numbers: ", selected_item_indices)
            print("Selected item costs: {}, total = {}".format(selected_costs, sum(selected_costs)))
            print("Selected item returns: {}, total = {}".format(selected_returns, sum(selected_returns)))
            print(selected_item_indices)
            print("Iter: {}".format(i))
    else:
        print("Problem too big to be solved with an Exact Sampler")
