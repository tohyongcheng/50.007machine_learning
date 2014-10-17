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

class Diabetes:
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

    for i in xrange(iterations):
      self.bgd()
      error = self.evaluation()

      self.weights.append(self.theta)
      self.errors.append(error)

      print "Iteration %s: BGD Error rate is %s" % (i+1,error)

  def run_stochastic_gradient_descent(self):
    iterations = 10000
    t = 0
    while(t != iterations):
      # randomise points
      if (t%self.n == 0):
        shuffle(self.indices) #shuffle only when its exceeded the number of data points

      idx = t%self.n
      self.sgd(idx)
      error = self.evaluation()

      if (t%100 == 0):
        self.weights.append(self.theta)
        self.errors.append(error)
        print "Iteration %s: SGD Error rate is %s" % (t+1,error)
      t += 1


  def sgd(self, i):
    gradient = [0]*21

    a = (- self.output[i]) /(1 + numpy.exp(self.output[i] * dp(self.theta, self.data[i])))
    for j in range(21):
      gradient[j] = a * self.data[i][j]

    new_theta = copy.deepcopy(self.theta)
    for j in range(len(new_theta)):
      new_theta[j] = new_theta[j] -  self.alpha * gradient[j]

    self.theta = new_theta



  def evaluation(self):
    s = 0
    for i in range(self.n):
      s += numpy.log(1 + numpy.exp(-self.output[i] * dp(self.theta, self.data[i])))
    s = (1.0/self.n) * s
    return s


def main():
  trainfile = argv[1]
  f1 = open(trainfile)
  data = []
  output = []
  for line in f1:
    d = line.split(",")
    for i in range(21):
      d[i] = float(d[i])

    data.append([1.0]+d[1:21])
    output.append(d[0])

  w1 = Diabetes(data,output)
  w1.run_stochastic_gradient_descent()


  print w1.errors
  print w1.weights

  fileout = open("output.txt", 'w')


  # write out to output
  for i in range(len(w1.errors)):
    s = ""
    for j in w1.weights[i]:
      s = s + str(j) + ","
    s = s + str(w1.errors[i])
    fileout.write(s+"\n")

  iterations = 100
  pylab.plot(numpy.arange(iterations), w1.errors, label="train")
  pylab.xlabel("Iterations")
  pylab.ylabel("Likelihood")
  pylab.legend(loc="upper left")
  pylab.show()

main()
