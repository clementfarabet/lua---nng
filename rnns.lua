-- Recurrent networks: flip-flop units and some common topologies
-- (Tom Schaul)

require 'nng'
require 'lab'

-- TODO: longer delays than 1
function nng.FlipFlop(size)
	local t = nng.Identity()
	t.io = {outputs={nng.Var(lab.zeros(size)), parent=t}}
	t.connect = function (self, inputs)
					assert(#inputs == 1)
				   	--table.insert(inputs[1].children, t)
				   	t.io.inputs = inputs   					
				end
	t.update = 	function (self)
					--if not self.io.inputs.valid then
					--	self.io.inputs.parent.update()
					--end
					self.io.outputs[1]:set(self.io.inputs[1])
				end
	return t
end



-- An Elman network has three fully connected layers (in, hidden, out),
-- with the activations of the hidden layer feeding back into the 
-- input, with a time-delay.

function elmanNodes(sizes, invar)
	assert (#sizes == 3)
	fflop = nng.FlipFlop(sizes[2])
	--m1 = nng.Linear(sizes[1]+sizes[2],sizes[2]){invar, fflop.io.outputs[1]}
	m1 = nng.Linear(sizes[1],sizes[2]){invar}
	m2 = nng.Tanh()(m1.io.outputs)
	fflop:connect(m2.io.outputs) -- connecting recurrent link
	m3 = nng.Linear(sizes[2],sizes[3])(m2.io.outputs)
	m4 = nng.Tanh()(m3.io.outputs)
	return fflop, m1, m2, m3, m4			
end

in1 = nng.Var()
g = nng.Graph(elmanNodes({1,2,3}, in1))

print(g, "init")
in1:set(lab.zeros(1))
print(g, "set")
g:update()
print(g, "go")
