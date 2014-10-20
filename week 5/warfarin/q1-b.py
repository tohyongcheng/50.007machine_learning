from sys import argv
import csv
import copy
import pylab
import numpy
from random import shuffle
from random import choice

def dp(a,b):
  result = 0
  for i in xrange(len(a)):
    result += a[i]*b[i]
  return result

class Warfarin:
  def __init__(self,data,output,theta = None, alpha = 0.1):
    self.data = data
    self.output = output
    self.n = len(data)
    self.alpha = alpha
    if theta == None:
      self.theta = [0]*21
    else:
      self.theta = theta
    self.weights = []
    self.errors = []
    self.indices = numpy.arange(self.n)

  def run_batch_gradient_descent(self):
    iterations = 100

    error = self.mse()
    print "Iteration 0: BGD Error rate is %s" % (error)

    for i in xrange(iterations):
      self.bgd()
      error = self.mse()

      self.weights.append(self.theta)
      self.errors.append(error)

      print "Iteration %s: BGD Error rate is %s" % (i+1,error)

  def run_stochastic_gradient_descent(self):
    iterations = 100

    error = self.mse()
    print "Iteration 0: SGD Error rate is %s" % (error)

    t = 0
    for i in range(iterations):      # this is 100
      shuffle(self.indices)
      self.sgd(self.indices[0])
      error = self.mse()

      self.weights.append(self.theta)
      self.errors.append(error)

      print "Iteration %s: SGD Error rate is %s" % (t+1,error)
      t += 1


  def bgd(self):
    right_sum = [0]*21

    for i in range(self.n):
      a = dp(self.theta,self.data[i]) - self.output[i]
      for j in range(21):
        right_sum[j] += a * self.data[i][j]

    new_theta = copy.deepcopy(self.theta)

    for i in range(len(new_theta)):
      new_theta[i] = new_theta[i] -  self.alpha * (1.0/self.n) * right_sum[i]

    self.theta = new_theta

  def sgd(self, i):
    right_sum = [0]*21

    a = dp(self.theta,self.data[i]) - self.output[i]
    for j in range(21):
      right_sum[j] += a * self.data[i][j]

    new_theta = copy.deepcopy(self.theta)
    for j in range(len(new_theta)):
      new_theta[j] = new_theta[j] -  self.alpha * right_sum[j]

    self.theta = new_theta



  def mse(self):
    s = 0
    for i in range(self.n):
      s += (  self.output[i] - dp(self.theta,self.data[i]) ) ** 2
    mse = (1.0/self.n) * s
    return mse


def main():
  trainfile, testfile, valfile = argv[1], argv[2], argv[3]
  f1 = open(trainfile)
  data = []
  output = []
  for line in f1:
    d = line.split(",")
    for i in range(21):
      d[i] = float(d[i])

    data.append([1.0]+d[1:21])
    output.append(d[0])

  f2 = open(testfile)
  test_data = []
  test_output = []
  for line in f2:
    d = line.split(",")
    for i in range(21):
      d[i] = float(d[i])

    test_data.append([1.0]+d[1:21])
    test_output.append(d[0])

  f3 = open(valfile)
  val_data = []
  val_output = []
  for line in f3:
    d = line.split(",")
    for i in range(21):
      d[i] = float(d[i])

    val_data.append([1.0]+d[1:21])
    val_output.append(d[0])



  w1 = Warfarin(data,output)

  # decide whether to run SGD or BGD
  w1.run_batch_gradient_descent()
  # w1.run_stochastic_gradient_descent()


  w2 = Warfarin(test_data,test_output)
  w3 = Warfarin(val_data,val_output)

  for i in range(len(w1.weights)):
    w2.theta = w1.weights[i]
    w2.errors.append(w2.mse())

    w3.theta = w1.weights[i]
    w3.errors.append(w3.mse())

  iterations = 100
  pylab.plot(numpy.arange(iterations), w1.errors, label="train")
  pylab.plot(numpy.arange(iterations), w2.errors, label="test")
  pylab.plot(numpy.arange(iterations), w3.errors, label="validation")
  pylab.xlabel("Iterations")
  pylab.ylabel("MSE")
  pylab.legend(loc="upper left")
  pylab.show()

main()
