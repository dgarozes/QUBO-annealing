import dimod
from math import log2, floor
import neal
from typing import List

import dwave.inspector
from dimod.reference.samplers import ExactSolver
from dwave.system import LeapHybridSampler
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler


class Qubo:

    """
    Class that contains a QUBO (Quadratic Unconstrained Boolean Optimization Problem)
    """

    # Number of binary variables (0/1 values)
    num_binary_variables = 0
    # Number of discrete variables (0, 1, 2... )values
    num_discrete_variables = 0
    num_constraints = 0

    def __init__(self,
                 variables: List,
                 values: List,
                 weights: List,
                 interactions: List,
                 capacity: int,
                 penalty: int,
                 sapi_token,
                 endpoint):

        if len(values) != len(weights) or len(values) != len(variables) or len(weights) != len(variables):
            print("Non-consistent values in the lists that contains the variables, values or weights.")
            raise ValueError

        if capacity <= 0:
            print("It is not feasible to solve the problem with a capacity equal or lower than zero.")
            raise ValueError

        self.variables = variables
        self.values = values
        self.weights = weights
        self.interactions = interactions
        self.capacity = capacity
        self.penalty = penalty

        self.lagrange = max(values)

        # Lucas's algorithm introduces additional slack variables to
        # handle the inequality. M+1 binary slack variables are needed to
        # represent the sum using a set of powers of 2.
        # * - Para una capacidad de 1000 unidades de carga, necesitaremos 9 + 1 variables
        self.M = floor(log2(capacity))
        self.num_slack_variables = self.M + 1

        self.num_binary_variables = len(variables)

        self.sapi_token = sapi_token
        self.endpoint = endpoint

        self.num_reads = 500
        self.chain_strength = 500000

        self.time_limit_lhs = 3

        self.bqm = None
        self.Q = None

        self.response_lhs = None
        self.response_qa = None
        self.response_sa = None
        self.response_exact_solver = None

        self.sample = None
        self.energy = None

        self.selected_item_indices = []
        self.stored_weight = None

    def build_qubo_bqm(self):
        """
        Created a QUBO assembled qith binary variables
        :return: Dictionary
        """
        # Initialize BQM - use large-capacity BQM so that the problem can be
        # scaled by the user.
        self.Q = {}
        self.bqm = dimod.AdjVectorBQM(dimod.Vartype.BINARY)
        # Slack variable list for Lucas's algorithm. The last variable has
        # a special value because it terminates the sequence.
        # * - Definimos las variables de holgura
        # * - sum(y) = weight_capacity
        y = [2 ** n for n in range(self.M)]
        y.append(self.capacity + 1 - 2 ** self.M)

        # Hamiltonian xi-xi terms
        # - * Lagrange * (peso^2 - valor)
        for index, (v, w) in enumerate(zip(self.values, self.weights)):
            key = (self.variables[index], self.variables[index])
            self.Q[key] = self.lagrange * (w ** 2) - v
            self.bqm.set_linear(self.variables[index], self.lagrange * (w ** 2) - v)

        # Hamiltonian xi-xj terms
        # Evitan que se seleccionen ítems de más
        for i in range(self.num_binary_variables):
            for j in range(i + 1, self.num_binary_variables):
                key = (self.variables[i], self.variables[j])
                weight1 = self.weights[i]
                weight2 = self.weights[j]
                self.Q[key] = 2 * self.lagrange * weight1 * weight2
                self.bqm.quadratic[key] = 2 * self.lagrange * weight1 * weight2

        # Constraints associated to items of the same owner:
        for constr in self.interactions:
            key = constr
            weight1 = self.weights[self.variables.index(constr[0])]
            weight2 = self.weights[self.variables.index(constr[1])]
            self.Q[key] = self.penalty * self.lagrange * weight1 * weight2
            self.bqm.quadratic[key] = self.penalty * self.lagrange * weight1 * weight2

        # Hamiltonian y-y terms
        for k in range(self.num_slack_variables):
            self.Q[('y' + str(k), 'y' + str(k))] = self.lagrange * (y[k] ** 2)
            self.bqm.set_linear('y' + str(k), self.lagrange * (y[k] ** 2))

        # Hamiltonian yi-yj terms
        for i in range(self.num_slack_variables):
            for j in range(i + 1, self.num_slack_variables):
                key = ('y' + str(i), 'y' + str(j))
                self.Q[key] = 2 * self.lagrange * y[i] * y[j]
                self.bqm.quadratic[key] = 2 * self.lagrange * y[i] * y[j]

        # Hamiltonian x-y terms
        for index, w in enumerate(self.weights):
            for j in range(self.num_slack_variables):
                key = (self.variables[index], 'y' + str(j))
                self.Q[key] = -2 * self.lagrange * w * y[j]
                self.bqm.quadratic[key] = -2 * self.lagrange * w * y[j]

    def run_leap_hybrid_solver(self):
        sampler = LeapHybridSampler()
        self.response_lhs = sampler.sample(self.bqm, time_limit=self.time_limit_lhs)

    def run_neal_exact_solver(self, limit_num_var=10):
        if len(self.variables) <= limit_num_var:
            sampler = ExactSolver()
            self.response_exact_solver = sampler.sample(self.bqm)
        else:
            print("Too many variables for the exact solver.")

    def run_simulated_annealing(self, sampleset=None):

        sampler = neal.SimulatedAnnealingSampler()
        self.response_sa = sampler.sample(self.bqm, num_reads=self.num_reads, sampleset=sampleset)

    def run_simultated_annealing_with_reheating_greedy(self,
                                                       num_iter=250,
                                                       sampleset=None):

        sampler = neal.SimulatedAnnealingSampler()
        self.response_sa = sampler.sample(self.bqm, num_reads=self.num_reads, sampleset=sampleset)

        best_sample = self.response_sa.first.sample
        best_energy = self.response_sa.first.energy

        for i in range(num_iter):
            if i % 10:
                print("Iteration : {iter}".format(iter=i))
                print("Best energy: {energy}".format(energy=best_energy))
            sampleset = sampler.sample(self.bqm, num_reads=self.num_reads, sampleset=best_sample)
            sample_iter = sampleset.first.sample
            energy = sampleset.first.energy
            print(energy)
            if energy < best_energy:
                best_energy = energy
                best_sample = sample_iter
                self.response_sa = sampleset

    def run_quantum_annealing(self):

        sampler = EmbeddingComposite(DWaveSampler(token=self.sapi_token, endpoint=self.endpoint))
        self.response_qa = sampler.sample_qubo(self.Q,
                                               num_reads=self.num_reads,
                                               chain_strength=self.chain_strength)

    def get_best_solution(self, algorithm):

        if algorithm == "ExactSolver" and self.response_exact_solver is not None:
            response = self.response_exact_solver
            self.sample = response.first.sample
            self.energy = response.first.energy
        elif algorithm == "SimulatedAnnealing" and self.response_sa is not None:
            response = self.response_sa
            self.sample = response.first.sample
            self.energy = response.first.energy
        elif algorithm == "QuantumAnnealing" and self.response_qa is not None:
            response = self.response_qa
            self.sample = response.first.sample
            self.energy = response.first.energy
        elif algorithm == "LeapHybridSolver" and self.response_lhs is not None:
            response = self.response_lhs
            self.sample = response.first.sample
            self.energy = response.first.energy
        else:
            print("Unknown solver or solution not available for the selected solver.")
            self.sample = None
            self.energy = None

    def get_selected_items(self):

        for varname, value in self.sample.items():
            # For each "x" variable, check whether its value is set, which
            # indicates that the corresponding item is included in the
            # knapsack
            if value and varname.startswith('x'):
                # The index into the weight array is retrieved from the
                # variable name
                self.selected_item_indices.append(int(varname[1:]))
        return self.selected_item_indices

    def get_weight_stored(self):

        if len(self.selected_item_indices) == 0:
            print("It is mandatory to calculate before the selected items")
            return None
        else:
            self.stored_weight = 0
            for ele in self.selected_item_indices:
                self.stored_weight += self.weights[self.variables.index("x" + str(ele))]

    def show_problem_inspection(self):

        if self.response_qa is not None:
            dwave.inspector.show(self.response_qa)
        else:
            print("Solution calculated with quantum annealing not available.")
            return None
