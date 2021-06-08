import dwave.inspector
from dwave.system.samplers import DWaveSampler
from dwave_qbsolv import QBSolv
from dwave.system.composites import EmbeddingComposite
import numpy as np

sapi_token = '#########'
endpoint = 'https://cloud.dwavesys.com/sapi'
#sampler = DWaveSampler(token=sapi_token, endpoint=endpoint)
sampler = EmbeddingComposite(DWaveSampler(token=sapi_token, endpoint=endpoint))
problem_size = 21
matrix = np.random.random((problem_size,problem_size))
matrix = (matrix + matrix.T)/2
Q = {}
for i in range(problem_size):
    for j in range(i, problem_size):
        if i==j:
            Q[(i,j)] = -problem_size/2
        else:
            Q[(i, j)] = matrix[i][j]

response = sampler.sample_qubo(Q,num_reads=1)

# Inspect
dwave.inspector.show(response)
