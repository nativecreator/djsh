#!/usr/bin/env python
# -*- coding: utf-8 -*-

# numpy 없이는 살 수가 없다.
import numpy

# Temporal Memory의 파이썬 구현
from nupic.research.temporal_memory import TemporalMemory as TM
from nupic.encoders.scalar import ScalarEncoder

# izip은 최대의 효율을 위해서 필요하다.
from itertools import izip as zip, count

import random
import matplotlib.pyplot as plt

# Unit 인코더 (범위: -100~100)
UnitEncoder= ScalarEncoder(21, -100, 100, False, 0, 0, 1, "unit")

EncodedUnitTable = {}

def CreateEncodedUnitTable():
    for i in range(-100, 101):
        EncodedUnitTable[i] = UnitEncoder.encode(i).tolist()

def DecodeUnit( U ):
    dic, name = UnitEncoder.decode(U)
    if len(name) > 0:
        return int(numpy.mean(dic['unit'][0]))

    return -999

# Vector 인코더
def EncodeVector( u, v ):
    U = EncodedUnitTable[ u ]
    V = EncodedUnitTable[ v ]

    return U + V

def DecodeVector( Vec ):
    EncodedWidth = UnitEncoder.getWidth();

    U = Vec[0           :EncodedWidth]
    V = Vec[EncodedWidth:EncodedWidth*2]

    u = numpy.asarray(U)
    v = numpy.asarray(V)

    return DecodeUnit(u), DecodeUnit(v)

# 일단 크기와 각도가 랜덤인 Vector Field(10 by 10)를 만들어보자.
CreateEncodedUnitTable()

xRange = 10
yRange = 10
colDims = UnitEncoder.getWidth() * 2 * xRange * yRange

tm = TM(columnDimensions=(colDims,),
        cellsPerColumn=4,
        initialPermanence=0.5,
        connectedPermanence=0.5,
        minThreshold=10,
        maxNewSynapseCount=20,
        permanenceIncrement=0.1,
        permanenceDecrement=0.0,
        activationThreshold=4,
        )

print "Started!"

#plt.ion()
plt.figure()
plt.title('Vector Field Prediction Using Machine Learning')

VectorSize = UnitEncoder.getWidth()*2

for i in range(10):
    Vec = numpy.zeros(xRange*yRange*VectorSize)
    z=0
    for x in range(xRange):
        for y in range(yRange):
            #V = EncodeVector(random.randint(-100, 100), random.randint(-100, 100))
            #u, v = DecodeVector(V)
            #print "(%d, %d) = v(%d, %d)" % (x, y, u, v)
            #plt.quiver(x, y, u, v, pivot='mid', scale=10, units='dots', width=1)
            Vec[z : z + UnitEncoder.getWidth()] = EncodedUnitTable[random.randint(-100, 100)]
            Vec[z + UnitEncoder.getWidth(): z + UnitEncoder.getWidth() * 2] = EncodedUnitTable[random.randint(-100, 100)]
            z = z + VectorSize

    activeColumns = set([j for j, k in zip(count(), Vec) if k == 1])
    tm.compute(activeColumns, learn = True)

    print "learned #%d" % i
    #plt.draw()
    #plt.pause(0.001)
    #plt.clf()
    #numpy.savetxt("Vec%d.txt" % i, Vec)

Vec = numpy.zeros(xRange * yRange * VectorSize)
z = 0
for x in range(xRange):
    for y in range(yRange):
        Vec[z: z + UnitEncoder.getWidth()] = EncodedUnitTable[random.randint(-100, 100)]
        Vec[z + UnitEncoder.getWidth(): z + UnitEncoder.getWidth() * 2] = EncodedUnitTable[random.randint(-100, 100)]
        z = z + VectorSize

activeColumns = set([j for j, k in zip(count(), Vec) if k == 1])
tm.compute(activeColumns, learn = False)

activeColumnsIndeces = [tm.columnForCell(i) for i in tm.getActiveCells()]
predictedColumnIndeces = [tm.columnForCell(i) for i in tm.getPredictiveCells()]

actColState = [1 if i in activeColumnsIndeces else 0 for i in range(tm.numberOfColumns())]
predColState = [1 if i in predictedColumnIndeces else 0 for i in range(tm.numberOfColumns())]

z = 0
for x in range(xRange):
    for y in range(yRange):
        AV = actColState[z: z + UnitEncoder.getWidth() * 2]
        PV = predColState[z: z + UnitEncoder.getWidth() * 2]
        PV = numpy.asarray( PV )
        u, v = DecodeVector( PV )
        if u != -999 and v != -999:
            print "(%d, %d) = v(%d, %d)" % (x, y, u, v)
            plt.quiver(x, y, u, v, pivot='mid', scale=10, units='dots', width=1)
        else:
            print "(%d, %d) = v(unpredictable)" % (x, y)
            plt.quiver(x, y, 100, 100, pivot='mid', scale=10, units='dots', width=1, color='r')
            plt.quiver(x, y, -100, 100, pivot='mid', scale=10, units='dots', width=1, color='r')
            plt.quiver(x, y, 100, -100, pivot='mid', scale=10, units='dots', width=1, color='r')
            plt.quiver(x, y, -100, -100, pivot='mid', scale=10, units='dots', width=1, color='r')
        #print "(%d, %d) = v(%d, %d)" % (x, y, u, v)
        #plt.quiver(x, y, u, v, pivot='mid', scale=10, units='dots', width=1)
        #plt.quiver(x, y, u, v, pivot='mid')
        z = z + UnitEncoder.getWidth() * 2

plt.show()

# 1

#Q = plt.quiver(U, V)
#qk = plt.quiverkey(Q, 0.5, 0.92, 2, r'$2 \frac{m}{s}$', labelpos='W',
#                   fontproperties={'weight': 'bold'})
#l, r, b, t = plt.axis()
#dx, dy = r - l, t - b
#plt.axis([l - 0.05*dx, r + 0.05*dx, b - 0.05*dy, t + 0.05*dy])

