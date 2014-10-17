from sys import argv
from random import shuffle
from random import choice
import csv

class Perceptron:
  def __init__(self,data,output):
    self.data = data
    self.output = output

  def dp(self,a,b):
    result = 0
    for i in xrange(len(a)):
      result += a[i]*b[i]
    return result

  def pla(self,w):
    test_w = w[:]
    indices = range(len(self.data))
    shuffle(indices)

    for i in indices:
      if self.evaluate(w, self.data[i],self.output[i]) == 1:
        # print "Running on data point %s" % (i)
        for j in xrange(len(self.data[i])):
          w[j] = w[j] + self.output[i] * self.data[i][j]
        break
    return w[:]

  def error_function(self,w):
    error = 0
    for i in xrange(len(self.data)):
      error += self.evaluate(w, self.data[i], self.output[i])
    return error

  def evaluate(self,w,x,output):
    # output is 1 or -1
    result = output * self.dp(w,x)
    if result <= 0:
      return 1
    else:
      return 0

  def run(self):
    iterations = 1000
    w = [0,0,0]
    best_w = [0,0,0]
    best_error = self.error_function(w)
    for i in xrange(iterations):
      prev_w = w[:]
      prev_error = self.error_function(prev_w)
      new_w = self.pla(prev_w)
      current_error = self.error_function(new_w)

      if current_error < best_error:
        best_w = new_w
        best_error = current_error
      w = new_w
      print "Iteration %d Current best error rate is %s, %s percent" % (i, best_error, 100-best_error*100.0/len(self.data))
      # print "."

    print "The best w is ", best_w
    print "Error rate for training is ", best_error
    return best_w


def main():
  trainfile, testfile = argv[1], argv[2]
  f = open(trainfile)
  data = []
  output = []

  for line in f:
    x1, x2, y = line.split(",")
    data.append([float(x1),float(x2),1])
    output.append(float(y))

  a = Perceptron(data,output)
  w = a.run()

  f2 = open(testfile)
  test_data = []
  test_output = []
  for line in f2:
    x1, x2, y = line.split(",")
    test_data.append([float(x1),float(x2),1])
    test_output.append(float(y))
  b = Perceptron(test_data,test_output)
  print "No. of errors for test case is", b.error_function(w)
  print "Accuracy for test case is", (100-b.error_function(w)*100.0/len(b.data))

main()
