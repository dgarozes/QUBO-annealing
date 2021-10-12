
from private.constants_private import endpoint, sapi_token
from qubo.qubo import Qubo


# Scenario 1
variables = ['x14', 'x15', 'x16', 'x17', 'x23', 'x24', 'x25', 'x26',
             'x27', 'x28', 'x31', 'x32', 'x34', 'x35', 'x36', 'x37',
             'x38', 'x40', 'x41', 'x42', 'x43', 'x44', 'x45', 'x46',
             'x5', 'x50', 'x51', 'x52', 'x53', 'x54', 'x55', 'x56',
             'x58', 'x6', 'x60', 'x61', 'x62', 'x63', 'x64', 'x65',
             'x67', 'x68', 'x69', 'x71', 'x72','x73', 'x77',  'x78',
             'x79', 'x82']

values = [9, 78, 70, 15, 1, 63, 101, 98, 83, 35, 6, 2, 21, 65, 75, 46, 11, 9, 79, 61, 49, 72, 84,
          14, 9, 7, 81, 90, 70, 83, 85, 29, 3, 6, 21, 69, 48, 9, 25, 11, 18, 36, 24, 29, 36, 11,
          4, 23, 8, 1]

weights = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
           5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

interactions = [('x46', 'x55'), ('x52', 'x63'), ('x52', 'x53'), ('x78', 'x79'), ('x51', 'x62'),
                ('x17', 'x27'), ('x61', 'x71'), ('x67', 'x78'), ('x53', 'x64'), ('x41', 'x52'),
                ('x63', 'x65'), ('x72', 'x82'), ('x53', 'x62'), ('x14', 'x5'), ('x42', 'x43'),
                ('x55', 'x64'), ('x54', 'x63'), ('x26', 'x37'), ('x16', 'x6'), ('x45', 'x54'),
                ('x63', 'x73'), ('x5', 'x6'), ('x43', 'x44'), ('x14', 'x25'), ('x35', 'x44'),
                ('x36', 'x45'), ('x61', 'x72'), ('x15', 'x5'), ('x71', 'x82'), ('x15', 'x16'),
                ('x69', 'x79'), ('x60', 'x72'), ('x72', 'x73'), ('x32', 'x41'), ('x16', 'x27'),
                ('x35', 'x46'), ('x58', 'x69'), ('x71', 'x72'), ('x17', 'x37'), ('x55', 'x65'),
                ('x31', 'x41'), ('x67', 'x69'), ('x41', 'x50'), ('x56', 'x65'), ('x50', 'x60'),
                ('x28', 'x37'), ('x53', 'x63'), ('x31', 'x42'), ('x23', 'x25'), ('x69', 'x78'),
                ('x43', 'x54'), ('x69', 'x77'), ('x27', 'x28'), ('x62', 'x63'), ('x58', 'x68'),
                ('x54', 'x64'), ('x50', 'x52'), ('x35', 'x36'), ('x45', 'x55'), ('x61', 'x62'),
                ('x32', 'x42'), ('x63', 'x64'), ('x35', 'x45'), ('x25', 'x26'), ('x44', 'x46'),
                ('x50', 'x61'), ('x52', 'x62'), ('x58', 'x78'), ('x44', 'x53'), ('x14', 'x24'),
                ('x67', 'x77'), ('x16', 'x5'), ('x60', 'x62'), ('x53', 'x55'), ('x23', 'x34'),
                ('x77', 'x78'), ('x42', 'x53'), ('x43', 'x53'), ('x77', 'x79'), ('x16', 'x26'),
                ('x54', 'x55'), ('x73', 'x82'), ('x54', 'x65'), ('x60', 'x71'), ('x71', 'x73'),
                ('x25', 'x35'), ('x46', 'x56'), ('x50', 'x51'), ('x34', 'x35'), ('x27', 'x37'),
                ('x31', 'x40'), ('x36', 'x37'), ('x23', 'x35'), ('x26', 'x35'), ('x67', 'x79'),
                ('x45', 'x56'), ('x51', 'x52'), ('x52', 'x61'), ('x68', 'x69'), ('x67', 'x68'),
                ('x23', 'x24'), ('x14', 'x35'), ('x27', 'x38'), ('x51', 'x61'), ('x25', 'x34'),
                ('x14', 'x34'), ('x17', 'x28'), ('x40', 'x51'), ('x58', 'x77'), ('x15', 'x23'),
                ('x15', 'x24'), ('x45', 'x46'), ('x26', 'x36'), ('x14', 'x23'), ('x41', 'x42'),
                ('x16', 'x25'), ('x15', 'x34'), ('x43', 'x52'), ('x44', 'x55'), ('x16', 'x17'),
                ('x15', 'x35'), ('x58', 'x67'), ('x53', 'x54'), ('x31', 'x32'), ('x25', 'x36'),
                ('x34', 'x45'), ('x15', 'x26'), ('x24', 'x25'), ('x17', 'x26'), ('x68', 'x79'),
                ('x40', 'x52'), ('x58', 'x79'), ('x68', 'x78'), ('x63', 'x72'), ('x64', 'x65'),
                ('x42', 'x51'), ('x60', 'x61'), ('x55', 'x56'), ('x27', 'x36'), ('x37', 'x46'),
                ('x24', 'x35'), ('x14', 'x15'), ('x62', 'x71'), ('x24', 'x34'), ('x26', 'x27'),
                ('x44', 'x45'), ('x34', 'x44'), ('x36', 'x46'), ('x55', 'x63'), ('x15', 'x6'),
                ('x40', 'x41'), ('x42', 'x50'), ('x37', 'x38'), ('x68', 'x77'), ('x15', 'x25'),
                ('x28', 'x38'), ('x62', 'x73'), ('x44', 'x54'), ('x42', 'x52'), ('x41', 'x51'),
                ('x51', 'x60'), ('x53', 'x65'), ('x62', 'x72'), ('x40', 'x42'), ('x40', 'x50'),
                ('x17', 'x38')]

penalty = 10
capacity = 100


qubo_p = Qubo(variables,
              values,
              weights,
              interactions,
              capacity,
              penalty,
              sapi_token,
              endpoint,
              "Scenario1")

qubo_p.build_qubo_bqm()

print("Simulated annealing")
qubo_p.run_simulated_annealing()
qubo_p.get_best_solution("SimulatedAnnealing")
print("Selected items")
print(qubo_p.get_selected_items())
qubo_p.get_weight_stored()
print("Weight stored: {w}".format(w=qubo_p.stored_weight))
print("Energy of the system: {energy}".format(energy=qubo_p.energy))

print("LeapHybridSolver")
qubo_p.run_leap_hybrid_solver()
qubo_p.get_best_solution("LeapHybridSolver")
print("Selected items")
print(qubo_p.get_selected_items())
qubo_p.get_weight_stored()
print("Weight stored: {w}".format(w=qubo_p.stored_weight))
print("Energy of the system: {energy}".format(energy=qubo_p.energy))

print("Simulated annealing")
qubo_p.run_simultated_annealing_with_reheating_greedy(num_iter=25)
qubo_p.get_best_solution("SimulatedAnnealing")
print("Selected items")
print(qubo_p.get_selected_items())
qubo_p.get_weight_stored()
print("Weight stored: {w}".format(w=qubo_p.stored_weight))
print("Energy of the system: {energy}".format(energy=qubo_p.energy))

print("Quantum annealing")
qubo_p.run_quantum_annealing()
qubo_p.get_best_solution("QuantumAnnealing")
print("Selected items")
print(qubo_p.get_selected_items())
qubo_p.get_weight_stored()
print("Weight stored: {w}".format(w=qubo_p.stored_weight))
print("Energy of the system: {energy}".format(energy=qubo_p.energy))
qubo_p.show_problem_inspection()
