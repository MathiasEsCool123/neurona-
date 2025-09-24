import numpy as np

class neuron:
    def __init__(self, n_input):
        self.weigth = np.random.randn(n_input)
        self.bias = np.random.randn()
        self.output = 0
        self.inputs = None
        self.dweigth = np.zeros_like(self.weigth)
        self.dbais = 0
            
    def activate(self, x):
        return 1/(1 + np.exp(-1))

    def derivate_activate(self,x):
        return x * (1-x)

    def foward(self, inputs):
        self.inputs = inputs 
        weighted_sum = np.dot(inputs, self.weigth) + self.bias 
        output = self.activate(weighted_sum)
        return self.output
    
    def backward(self, d_output, learning_learn):
        d_activation = d_output *self.derivate_activate(self.output)
        self.dweigth = np.dot(self.inputs,d_activation)
        self.dbais = d_activation
        self.weigth -= self.dweigth *learning_rate
        d_input = np.dot(d_activation , self.weigth)
        self.bias -= learning_rate * self.dbais
        return d_input

if __name__ == "__main__":
    neuron = neuron(3)
    inputs = np.array([1,2,3])
    output = neuron.foward(inputs)
    print("Neuron output:",output)