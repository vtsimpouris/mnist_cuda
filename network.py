"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random
from threading import Lock
import Queue
import pycuda
import pycuda.driver as drv
import pycuda.autoinit
import threading
from pycuda.compiler import SourceModule
import timeit
# Third-party libraries
import numpy as np
from operator import add
'''Backprop has now become an independant class so that it can run
as a sepaerate thread each it is called,this results in parallelizing
a part of the code below'''
class backprop(threading.Thread):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
	def __init__(self,x,y,weights,biases,delta_nabla_b,delta_nabla_w,queue):
	    threading.Thread.__init__(self)
	    self.x = x
	    self.y = y
	    self.weights = weights
	    self.biases = biases
	    self.delta_nabla_b = delta_nabla_b
	    self.delta_nabla_w = delta_nabla_w
	    self.queue = queue
	def run(self):
	    #initialize the cuda device, in my version numer 0
	    self.dev = drv.Device(0)
            self.ctx = self.dev.make_context()

            self.delta_nabla_b = [np.zeros(b.shape) for b in self.biases]
            self.delta_nabla_w = [np.zeros(w.shape) for w in self.weights]
          # feedforward
            activation = self.x
            activations = [self.x] # list to store all the activations, layer by layer
            zs = [] # list to store all the z vectors, layer by layer
            for b, w in zip(self.biases, self.weights):
                z = np.dot(w, activation)+b
                zs.append(z)
                activation = sigmoid(z)
                activations.append(activation)
          # backward pass    
            delta = cost_derivative(activations[-1], self.y) * \
                sigmoid_prime(zs[-1])
            self.delta_nabla_b[-1] = delta
            self.delta_nabla_w[-1] = np.dot(delta, activations[-2].transpose())
            # Note that the variable l in the loop below is used a little
            # differently to the notation in Chapter 2 of the book.  Here,
            # l = 1 means the last layer of neurons, l = 2 is the
            # second-last layer, and so on.  It's a renumbering of the
            # scheme in the book, used here to take advantage of the fact
            # that Python can use negative indices in lists.    
            for l in xrange(2, 3):
                z = zs[-l]
                sp = sigmoid_prime(z)
		M = np.int32(self.weights[0][0].shape)
	        N = len(self.weights[0])
		n = np.int32(N)
		mutex = Lock()

		drv.init()
		'''This kernel is the cuda version of this line:
		delta = np.dot(self.weights[-l+1].transpose(), delta) * sp'''

		'''!!Note that numpy is already running in parallel(4-8 threads), so the kernel
		is 15-20% better than the numpy version and only for big matrices'''

	    	mod = SourceModule("""
	    	__global__ void nablabdot(float *sp,float *w, float *delta,float *nabla_b,int N){
		int j;
		int i = threadIdx.x + blockIdx.x*blockDim.x;
		nabla_b[i] = 0;
		for(j = 0;j < 10;j++){
			nabla_b[i] = w[i+j*N]*delta[j] + nabla_b[i];
		}

		nabla_b[i] = nabla_b[i]*sp[i];
		
            	 }
            	""")
	    	func2 = mod.get_function("nablabdot")
	    	nabla = self.delta_nabla_b[-l].astype(np.float32)#a temp matrix is needed to store the cuda kernel results
	    	func2(drv.In(sp.astype(np.float32)), drv.In(self.weights[-l+1].astype(np.float32)),drv.In(delta.astype(np.float32)), 
		drv.Out(nabla),n, block=(100,1,1), grid=(N/100,1,1))
	    	self.delta_nabla_b[-l] = nabla
		
	        
	        '''This kernel is the cuda version of this line:
		nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())'''

		'''This matrix multipication can be done in one move in cuda,
		but still the kernel is only around 20% faster than numpy version, because
		of big memmory overheads'''
 
		mod = SourceModule("""
		__global__ void nablawdot(float *delta,float *a, float *nabla,int M){
		int j = threadIdx.y + blockIdx.y*blockDim.y;
		int i = threadIdx.x + blockIdx.x*blockDim.x;
		
		
			nabla[i*M+j] = delta[i]*a[j];
		
             	}
        	""")
		func = mod.get_function("nablawdot")
	   
	        nabla = self.delta_nabla_w[-l].astype(np.float32)#a temp matrix is needed to store the cuda kernel results
	        func(drv.In(delta.astype(np.float32)), drv.In(activations[-l-1].astype(np.float32)), drv.Out(nabla), M, block=(10,28,1), grid=(N/10,28,1))
	        self.ctx.pop()
	        self.delta_nabla_w[-l] = nabla	    
	        del self.ctx
				
		 
		'''sync the queue so that nabla_b is not stored
		in nabla_w place and vice versa.'''

				
		mutex.acquire()
		self.queue.put(self.delta_nabla_w)
		self.queue.put(self.delta_nabla_b)	
		mutex.release()
	 

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
	
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
	delta_nabla_b = [np.zeros(b.shape) for b in self.biases]
        delta_nabla_w = [np.zeros(w.shape) for w in self.weights]
	gpu_thread_list = []
	queue = Queue.Queue()
	'''All x,y belonging to a mini batch are uncorelated between them,
	so each backpropagation instance to update the nablas is becoming a thread, so that N 
	backprop threads are running in parallel''' 
	start = timeit.default_timer()
        for x, y in mini_batch:
	    gpu_thread = backprop(x,y,self.weights,self.biases,nabla_b,nabla_w,queue)
    	    gpu_thread.start() 
	    gpu_thread_list.append(gpu_thread)
	'''join threads that are still running, and recover the
	new nabla values'''
	'''I used map instead of zip to add lists because
	it is much faster'''  
	for i in xrange(0,len(mini_batch)):
	    gpu_thread.join()
	    delta_nabla_w = queue.get()
	    delta_nabla_b = queue.get()
	    map(add, nabla_w, delta_nabla_w)
	    map(add, nabla_b, delta_nabla_b)    
	#print time elapsed to update a mini batch              
	stop = timeit.default_timer()
	print stop - start
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]



    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

def cost_derivative(output_activations, y):
     """Return the vector of partial derivatives \partial C_x /
     \partial a for the output activations."""
     return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
