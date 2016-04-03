#!/usr/bin/env python
# -*- coding: utf-8 -*-

# numpy 없이는 살 수가 없다.
import numpy

# Temporal Memory의 파이썬 구현
from nupic.research.temporal_memory import TemporalMemory as TM
from nupic.encoders.scalar import ScalarEncoder

import random
import matplotlib.pyplot as plt

# Unit 인코더 (범위: -100~100)
UnitEncoder= ScalarEncoder(21, -100, 100, False, 0, 0, 1, "unit")

EncodedUnitTable = {}

def CreateEncodedUnitTable():
    for i in range(-100, 101):
        EncodedUnitTable[i] = UnitEncoder.encode(i).tolist()

def DecodeUnit( U ):
    return int(float( UnitEncoder.decode(U)[0]['unit'][1]))

# Vector 인코더
def EncodeVector( u, v ):
    U = EncodedUnitTable[ u ]
    V = EncodedUnitTable[ v ]
    #return numpy.append(U, V)
    return U + V

def DecodeVector( Vec ):
    U = Vec[0:UnitEncoder.getWidth()]
    V = Vec[UnitEncoder.getWidth():UnitEncoder.getWidth() + UnitEncoder.getWidth()]

    #return DecodeUnit(U), DecodeUnit(V)
    u = -999
    v = -999

    for item in EncodedUnitTable.items():
        if item[1] == U:
            u = item[0]
        if item[1] == V:
            v = item[0]

    return u, v

# 일단 크기와 각도가 랜덤인 Vector Field(10 by 10)를 만들어보자.
VectorField = numpy.zeros((0,), dtype=numpy.uint8)

CreateEncodedUnitTable()

print "Started!"

plt.ion()
plt.figure()
plt.title('Vector Field Prediction Using Machine Learning')
plt.show()

for i in range(10):
    for x in range(20):
        for y in range(20):
            V = EncodeVector(random.randint(-100, 100), random.randint(-100, 100))
            u, v = DecodeVector(V)
            #print "(%d, %d) = v(%d, %d)" % (x, y, u, v)
            plt.quiver(x, y, u, v, pivot='mid', scale=10, units='dots', width=1)
        #VectorField = numpy.append( VectorField, V )
    plt.draw()
    plt.pause(0.001)
    plt.clf()
#numpy.savetxt('file.txt', VectorField)


# 1

#Q = plt.quiver(U, V)
#qk = plt.quiverkey(Q, 0.5, 0.92, 2, r'$2 \frac{m}{s}$', labelpos='W',
#                   fontproperties={'weight': 'bold'})
#l, r, b, t = plt.axis()
#dx, dy = r - l, t - b
#plt.axis([l - 0.05*dx, r + 0.05*dx, b - 0.05*dy, t + 0.05*dy])

