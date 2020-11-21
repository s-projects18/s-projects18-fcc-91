import warnings
warnings.filterwarnings('error')

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    r = False
    arg = -x
    arg[arg > 100] = 100
    arg[arg< -100] = -100
    r = 1/(1+np.exp(arg))
    return r

# Training
def train(X, y, weight1=False, weight2=False, iterations=600000, debug=False):
    num_input_nodes    = X.shape[1]
    num_test = X.shape[0]
    num_output_nodes   = y.shape[1]
    
    if type(weight1)==bool and weight1==False:
        np.random.seed(1)
        weight1 = 2*np.random.random((num_input_nodes,num_test)) - 1
        weight2 = 2*np.random.random((num_test, num_output_nodes)) - 1

    l0 = X
    for j in range(iterations):
        # calculate outputs based on weights
        l1 = nonlin(np.dot(l0,weight1)) 
        l2 = nonlin(np.dot(l1,weight2))

        # l2_error comes first, then l1_error => backpropagation!
        # l2_error
        l2_error = y - l2 
        if debug:
          if (j% 10000) == 0: # debug
            print("Error:" + str(np.mean(np.abs(l2_error))) )
        l2_delta = l2_error * nonlin(l2,deriv=True)

        # l1_error
        l1_error = l2_delta.dot(weight2.T) 
        l1_delta = l1_error * nonlin(l1,deriv=True)

        # adjust weights
        weight2 += l1.T.dot(l2_delta)
        weight1 += l0.T.dot(l1_delta)

    if debug:
      print("Output After Training:", l2)         
    return (weight1, weight2)


# prediction
def predict(weights, node):
    out = node
    for weight in weights:
        out = nonlin(np.dot(out,weight))
    return out 

# data container for X,y
class Batches:
    '''
    Object that stores 2 numpy-arrays: x and y
    '''
    def __init__(self):
        self.x = False
        self.y = False

    def addRows(self, dx, dy):
        '''
        dx and dy must be n-dimensional lists<br>
        eg: addRows([[1,1]], [[1]])
        '''
        if(self.x is False or self.y is False):
            self.x = np.array(dx)
            self.y = np.array(dy)
        else:
            self.x = np.append(self.x, dx, axis=0)
            self.y = np.append(self.y, dy, axis=0)
    
    def getData(self):
        '''
        return (x, y)
        '''
        return (self.x, self.y)


# print several weight matrices in one line
import numpy as np

def s_weights(m, width):
    f = '{: >'+str(width)+'}'
    a = '|'
    z = ' |'
    l = []
    for row in range(len(m)):
        t = a
        for col in range(len(m[row])):
            t += f.format(round(m[row][col],2))
        t += z
        l.append(t)
    return l

# arr: array of matrices printed in 1 line
def p_weights(arr, width=6):
    l = []
    sep = ' '
    num_rows = 0

    # single np-array -> [arr]
    if(type(arr) is np.ndarray):
        t = []
        t.append(arr)
        arr = t

    for a in arr: 
        # vector -> matrix
        if len(a.shape)==1:
            a = np.array([a.tolist()]).T

        sw = s_weights(a, width)
        if( len(sw)>num_rows ):
            num_rows = len(sw)
        l.append(sw)
  
    ln = [0] * len(l)
    for i in range(num_rows):
        t = ''
        
        for j in range(len(l)):
            line = ''
            try:
                line += l[j][i] + sep
                ln[j] = len(line)
            except:
                fr = '{: >'+str(ln[j])+'}'
                line += fr.format('')
            t += line
        print(t)


