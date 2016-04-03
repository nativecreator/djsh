#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""A simple program that demonstrates the working of the spatial pooler"""
"""spatial pooler의 동작을 보여주는 작은 프로그램"""

import numpy as np
import random
from nupic.bindings.algorithms import SpatialPooler as SP



uintType = "uint32"



class Example(object):
  """A class to hold our code.
  TODO: Get rid of this class, it just makes it more difficult to read the
  code.
  """


  def __init__(self, inputDimensions, columnDimensions):
    """
     Parameters:
     ----------
     _inputDimensions: The size of the input. (m,n) will give a size m x n
     _inputDimensions: 입력의 크기. (m, n) 이라면 m x n 이 된다.
     _columnDimensions: The size of the 2 dimensional array of columns
     _columnDimensions: 2차원 컬럼 배열의 크기
     """
    self.inputDimensions = inputDimensions
    self.columnDimensions = columnDimensions
    self.inputSize = np.array(inputDimensions).prod()
    self.columnNumber = np.array(columnDimensions).prod()
    self.inputArray = np.zeros(self.inputSize, dtype=uintType)
    self.activeArray = np.zeros(self.columnNumber, dtype=uintType)

    random.seed(1)

    self.sp = SP(self.inputDimensions,
                 self.columnDimensions,
                 potentialRadius = self.inputSize,
                 numActiveColumnsPerInhArea = int(0.02*self.columnNumber),
                 globalInhibition = True,
                 seed = 1,
                 synPermActiveInc = 0.01,
                 synPermInactiveDec = 0.008)


  def createInput(self):
    """create a random input vector"""

    print "-" * 70 + "Creating a random input vector" + "-" * 70

    #clear the inputArray to zero before creating a new input vector
    self.inputArray[0:] = 0

    for i in range(self.inputSize):
      #randrange returns 0 or 1
      #0이나 1로 랜덤하게 집어넣는다.
      self.inputArray[i] = random.randrange(2)


  def run(self):
    """Run the spatial pooler with the input vector
       input vector로 sparial pooler를 돌려보자"""

    print "-" * 80 + "Computing the SDR" + "-" * 80

    #activeArray[column]=1 if column is active after spatial pooling
    #activeArray[column]=1 이면 spatial pooling 결과 활성화된 컬럼이다.
    self.sp.compute(self.inputArray, True, self.activeArray)

    print self.activeArray.nonzero()


  def addNoise(self, noiseLevel):
    """Flip the value of 10% of input bits (add noise)
    :param noiseLevel: The percentage of total input bits that should be flipped
       10% 정도 input bits를 뒤집어서(flip) 노이즈를 추가하자
    :param noiseLevel: 몇 퍼센트나 노이즈를 추가할 것인지?
    """

    for _ in range(int(noiseLevel * self.inputSize)):
      # 0.1*self.inputSize represents 10% of the total input bits
      # random.random() returns a float between 0 and 1
      # 0.1*self.inputSize 이면 전체 input bits의 10%다.
      # random.random() 은 0과 1사이의 실수를 반환하다.
      randomPosition = int(random.random() * self.inputSize)

      # Flipping the bit at the randomly picked position
      # 랜덤한 위치의 bit를 뒤집자... 1이면 0으로, 0이면 1로!
      if self.inputArray[randomPosition] == 1:
        self.inputArray[randomPosition] = 0

      else:
        self.inputArray[randomPosition] = 1

      # Uncomment the following line to know which positions had been flipped.
      # 아래 주석을 풀면, 어떤 위치가 뒤집어진 건지 알 수 있다.
      # print "The value at " + str(randomPosition) + " has been flipped"

example = Example((32, 32), (64, 64))

# Lesson 1
print "\n \nFollowing columns represent the SDR"
print "Different set of columns each time since we randomize the input"
print "Lesson - different input vectors give different SDRs\n\n"

# Trying random vectors
for i in range(3):
  example.createInput()
  example.run()

# Lesson 2
print "\n\nIdentical SDRs because we give identical inputs"
print "Lesson - identical inputs give identical SDRs\n\n"

print "-" * 75 + "Using identical input vectors" + "-" * 75

# Trying identical vectors
for i in range(2):
  example.run()

# Lesson 3
print "\n\nNow we are changing the input vector slightly."
print "We change a small percentage of 1s to 0s and 0s to 1s."
print "The resulting SDRs are similar, but not identical to the original SDR"
print "Lesson - Similar input vectors give similar SDRs\n\n"

# Adding 10% noise to the input vector
# Notice how the output SDR hardly changes at all
print "-" * 75 + "After adding 10% noise to the input vector" + "-" * 75
example.addNoise(0.1)
example.run()

# Adding another 20% noise to the already modified input vector
# The output SDR should differ considerably from that of the previous output
print "-" * 75 + "After adding another 20% noise to the input vector" + "-" * 75
example.addNoise(0.2)
example.run()
