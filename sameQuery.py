# _*_ coding: utf-8 _*_
from nltk.metrics import *

# 相似性度量
training='PERSON OTHER PERSON OTHER OTHER ORGANIZATION'.split()
testing='PERSON OTHER OTHER OTHER OTHER OTHER'.split()
print(accuracy(training, testing))

trainset = set(training)
testset = set(testing)
print(precision(trainset, testset))

print(recall(trainset, testset))
print(f_measure(trainset, testset))

# 使用编辑距离算法执行相似性度量
import nltk
from nltk.metrics import *

print(edit_distance("relate", "relation"))
print(edit_distance("suggestion", "calculation"))

# 使用jaccard系数执行相似性度量
import nltk
from nltk.metrics import jaccard_distance
x = set([10, 20, 30, 40])
y = set([20, 30, 60])
print(jaccard_distance(x, y))
