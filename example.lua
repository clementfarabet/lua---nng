
-- load nng
require 'nng'

-- declare an input variable
input1 = nng.Var()

-- declare a couple of nodes and their connections
mod1 = nng.Linear(10,10){input1}
mod2 = nng.Tanh()(mod1.io.outputs)

-- name output
output1 = mod2.io.outputs[1]

-- create a graph on the nodes
g = nng.Graph(mod1,mod2)

-- update graph
print('time 0')
print(g)

print('time 1 (set inputs)')
input1:set(lab.randn(10))
print(g)

print('time 2 (updated graph)')
g:update(output1)
print(g)
