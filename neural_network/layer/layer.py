import numpy as np

from neuron.neuron import Neuron

class layer:
    def __init__(self, num_numeros , inputs_size):
        self.neurons = [Neuron(inputs_size)for _ in range (num_numeros)]
    
    def forward (self, inputs):
        return np.array([neuron.forward(inputs)for neuron in self.neuron])
    def backward(self, d_outputs, learning_rate):
        d_inputs = np.zeros(self.neuron[0].inputs)
        for i,  neuron in enumerate(self.neurons):
            d_inputs += neuron.backward(d_outputs[i], learning_rate)
        return d_inputs
    
if __name__ =="__main__":
    layer = layer(3 , 4)
    inputs = np.array ([1 , 8  , 5 , 6])
    
    layer_output = layer.forward(inputs)
    print("layer outputs:", layer_output)