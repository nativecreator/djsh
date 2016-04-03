#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
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


print """
This program shows how to access the Temporal Memory directly by demonstrating
이 프로그램은 Temporal Memory에 직접 접근하는 방법을 데몬스트레이션으로 보여준다.

how to create a TM instance, train it with vectors, get predictions, and
inspect the state.
어떻게 TM 인스턴스를 생성하는지, vector들을 학습시키고, 예측하는지, 스테이트를 점검하는지.

The code here runs a very simple version of sequence learning, with one
cell per column. The TP is trained with the simple sequence A->B->C->D->E
이 코드는 컬럼당 하나의 셀을 갖는 시퀀스 러닝의 매우 간단한 버전이다.
TP는 매우 간단한 시퀀스인 A->B->C->D->E 를 훈련한다.

HOMEWORK: once you have understood exactly what is going on here, try changing
cellsPerColumn to 4. What is the difference between once cell per column and 4
cells per column?
숙제: 뭐가 어떻게 돌아가는지 정확히 이해했다면, 컬럼당 셀을 4개로 바꿔봐라.
컬럼당 하나의 셀인 경우와 4개인 경우가 어떻게 다른가?

PLEASE READ THROUGH THE CODE COMMENTS - THEY EXPLAIN THE OUTPUT IN DETAIL

"""

# Can't live without numpy
# numpy 없이는 살 수가 없다.
import numpy

# izip for maximum efficiency
# izip은 최대의 효율을 위해서 필요하다.
from itertools import izip as zip, count

# Python implementation of Temporal Memory
# Temporal Memory의 파이썬 구현
from nupic.research.temporal_memory import TemporalMemory as TM


# Utility routine for printing the input vector
# input vector를 출력하기 위한 유틸리티 루틴
def formatRow(x):
  s = ''
  for c in range(len(x)):
    if c > 0 and c % 10 == 0:
      s += ' '
    s += str(x[c])
  s += ' '
  return s


# Step 1: create Temporal Pooler instance with appropriate parameters
# Step 1: 적절한 파라미터들로 Temporal Pooler 인스턴스를 만든다.
tm = TM(columnDimensions = (50,),
        cellsPerColumn=2,
        initialPermanence=0.5,
        connectedPermanence=0.5,
        minThreshold=10,
        maxNewSynapseCount=20,
        permanenceIncrement=0.1,
        permanenceDecrement=0.0,
        activationThreshold=8,
        )


# Step 2: create input vectors to feed to the temporal pooler. Each input vector
# must be numberOfCols wide. Here we create a simple sequence of 5 vectors
# representing the sequence A -> B -> C -> D -> E
# Step 2: temporal pooler에 넣을 input vector들을 만든다.
# 각 input vector들은 반드시 numberOfCols 너비와 같아야 한다.
# 우리는 A -> B -> C -> E 의 시퀀스를 표현하는 5개의 간단한 vector들을 만들 것이다.
x = numpy.zeros((5, tm.numberOfColumns()), dtype="uint32")
x[0, 0:10] = 1    # Input SDR representing "A", corresponding to columns 0-9
x[1, 10:20] = 1   # Input SDR representing "B", corresponding to columns 10-19
x[2, 20:30] = 1   # Input SDR representing "C", corresponding to columns 20-29
x[3, 30:40] = 1   # Input SDR representing "D", corresponding to columns 30-39
x[4, 40:50] = 1   # Input SDR representing "E", corresponding to columns 40-49


# Step 3: send this simple sequence to the temporal memory for learning
# We repeat the sequence 10 times
# Step 3: 이 간단한 시퀀스를 temporal memory에 보내서 학습을 하게 한다.
# 우리는 이 시퀀스를 10번 반복해서 학습하게 한다.
for i in range(10):

  # Send each letter in the sequence in order
  # 시퀀스의 각 문자를 순서대로 보낸다.
  for j in range(5):
    xj = x[j]

    activeColumns = set([k for k, l in zip(count(), x[j]) if l == 1])

    # The compute method performs one step of learning and/or inference. Note:
    # here we just perform learning but you can perform prediction/inference and
    # learning in the same step if you want (online learning).
    # compute 메소드는 학습이나 추론의 한 단계를 수행한다.
    # 여기서는 단지 학습을 위해서 수행하지만, 원한다면 같은 방법으로 예측/추론을 위해서 수행할 수 있다.
    tm.compute(activeColumns, learn = True)

    # The following print statements can be ignored.
    # Useful for tracing internal states
    # 아래는 무시해도 되는 상태 출력이다.
    # 내부상태를 추적할 때 유용하다.
    print("active cells " + str(tm.getActiveCells()))
    print("predictive cells " + str(tm.getPredictiveCells()))
    print("winner cells " + str(tm.getWinnerCells()))
    print("# of active segments " + str(tm.connections.numSegments()))

  # The reset command tells the TP that a sequence just ended and essentially
  # zeros out all the states. It is not strictly necessary but it's a bit
  # messier without resets, and the TP learns quicker with resets.
  # reset 명령은 TP에게 시퀀스가 끝다는 걸 알리고, 모든 상태를 0으로 되돌린다.
  # 이게 꼭 필요한 건 아니지만, reset을 하지 않으면 좀 지저분하고, TP가 reset을 할 때 TP가 더 빨리 배운다.
  tm.reset()


#######################################################################
#
# Step 3: send the same sequence of vectors and look at predictions made by
# temporal memory
# Step 3: 동일한 시퀀스 vector들을 보내보고, temporal memory가 어떻게 예측하는지 보자.
for j in range(5):
  print "\n\n--------","ABCDE"[j],"-----------"
  print "Raw input vector : " + formatRow(x[j])
  activeColumns = set([i for i, j in zip(count(), x[j]) if j == 1])
  # Send each vector to the TM, with learning turned off
  # 학습을 끈 상태에서, 각 vector들을 TM으로 보내본다.
  tm.compute(activeColumns, learn = False)

  # The following print statements prints out the active cells, predictive
  # cells, active segments and winner cells.
  # 아래의 출력문에서는 활성화된 셀들, 예측된 셀들, 활성화된 segments와 승리한 셀들을 출력한다.
  #
  # What you should notice is that the columns where active state is 1
  # represent the SDR for the current input pattern and the columns where
  # predicted state is 1 represent the SDR for the next expected pattern
  # 주목해야할 것은 active state의 입력패턴 SDR을 1로 표상하고 있는 컬럼들과
  # 다음에 기대되는 패턴 SDR을 1로 표상하고 있는 predicted state의 컬럼들이다.
  print "\nAll the active and predicted cells:"

  print("active cells " + str(tm.getActiveCells()))
  print("predictive cells " + str(tm.getPredictiveCells()))
  print("winner cells " + str(tm.getWinnerCells()))
  print("# of active segments " + str(tm.connections.numSegments()))

  activeColumnsIndeces = [tm.columnForCell(i) for i in tm.getActiveCells()]
  predictedColumnIndeces = [tm.columnForCell(i) for i in tm.getPredictiveCells()]


  # Reconstructing the active and inactive columns with 1 as active and 0 as
  # inactive representation.
  # 활성화되거나 비활성화된 컬럼들을 1과 0을 이용하여 재구성한다.

  actColState = ['1' if i in activeColumnsIndeces else '0' for i in range(tm.numberOfColumns())]
  actColStr = ("".join(actColState))
  predColState = ['1' if i in predictedColumnIndeces else '0' for i in range(tm.numberOfColumns())]
  predColStr = ("".join(predColState))

  # For convenience the cells are grouped
  # 10 at a time. When there are multiple cells per column the printout
  # is arranged so the cells in a column are stacked together
  # 편의상 한 번에 10개의 셀들을 그룹화 했다.
  # 컬럼당 복수의 셀들이 존재할 때는 스택을 함께 쌓아서 보여준다.
  print "Active columns:    " + formatRow(actColStr)
  print "Predicted columns: " + formatRow(predColStr)

  # predictedCells[c][i] represents the state of the i'th cell in the c'th
  # column. To see if a column is predicted, we can simply take the OR
  # across all the cells in that column. In numpy we can do this by taking
  # the max along axis 1.
  # predictedCells[c][i] 는 c번째 컬럼의 i번 째 셀을 나타낸다.
  # 어떤 컬럼이 예측되었는지 보기 위해, 그 컬럼 안의 모든 셀에 대해 OR연산을 취해볼 수 있다.
  # numpy를 이용하면, 축 1에서 최대값만 취하면 된다.
