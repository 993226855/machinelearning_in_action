import itertools
#生成候选集1-项集
def createC1(dataSet):
    C1=set(itertools.chain(*dataSet))
    return [frozenset([i]) for i in C1]
#按条件过滤，生成频繁1-项集
def scanD(dataSet,Ck,min_support):
    support={}
    for i in dataSet:
        for j in Ck:
            if j.issubset(i):
                support[j]=support.get(j,0)+1
    n=len(dataSet)
    return {key:val/n for key,val in support.items() if val/n>=min_support}
#通过k-项集构造(k+1)-项集
def aprioriGen(Lk):
    #频繁项集的个数
    lenLk=len(Lk)
    #一个频繁项集元素的个数
    k=len(Lk[0])
    if lenLk>1 and k>0:
        return set([Lk[i].union(Lk[j])
            for i in range(lenLk-1)
            for j in range(i+1,lenLk)
            if len(Lk[i] | Lk[j])==k+1])

# 整合前面的小函数构建一个可循环的构建k+1-项集
def apriori(dataSet, min_support=0.5):
    C1 = createC1(dataSet)
    L1 = scanD(dataSet, C1, min_support)
    L = [L1]
    k = 2
    while len(L[k - 2]) > 1:
        Ck = aprioriGen(list(L[k - 2].keys()))
        Lk = scanD(dataSet, Ck, min_support)
        if len(Lk) > 0:
            L.append(Lk)
            k += 1
        else:
            break
    d = {}
    for Lk in L:
        d.update(Lk)
    return d

#生成规则LHS-->RHS
def rulesGen(iterable):
    subSet=[]
    for i in range(1,len(iterable)):
        subSet.extend(itertools.combinations(iterable,i))
    return [(frozenset(lhs),frozenset(iterable.difference(lhs)))
           for lhs in subSet]

import pandas as pd
def arules(dataSet, min_support=0.5):
    L = apriori(dataSet, min_support)
    rules = []
    for Lk in L.keys():
        if len(Lk) > 1:
            rules.extend(rulesGen(Lk))
    scl = []
    for rule in rules:
        lhs = rule[0];
        rhs = rule[1]
        support = L[lhs | rhs]
        confidence = support / L[lhs]
        lift = confidence / L[rhs]
        scl.append({'LHS': lhs, 'RHS': rhs, 'support': support, 'confidence': confidence, 'lift': lift})
    return pd.DataFrame(scl)

dataSet=[['A','C','D'],
        ['B','C','E'],
        ['A','B','C','E'],
         ['B','E']]
# C1=createC1(dataSet)
# min_support=0.4
# L1=scanD(dataSet,C1,min_support=min_support)
# C2=aprioriGen(list(L1.keys()))
# L2=scanD(dataSet,C2,min_support)

d=apriori(dataSet,min_support=0.4)
ss=frozenset(['A','B','C'])
rulesGen(ss)
res=arules(dataSet,0.4)
print(res.head(10))

