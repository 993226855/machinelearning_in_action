# -*- coding: utf-8 -*-
import os
from jieba import posseg as pseg
import pandas as pd

def getP(word,NBDict,flag):

	if(word in NBDict):
		return NBDict[word][flag]+1
	else:	
		return 1

def classify(wordsTest,NBDict):
	p0=p1=p2=1
	for words in wordsTest:
		print(words[0])
		print(p0,p1,p2)
		p0*=getP(words[0],NBDict,0)/(NBDict['evalation'][0]+len(wordsTest))
		p1*=getP(words[0],NBDict,1)/(NBDict['evalation'][1]+len(wordsTest))
		p2*=getP(words[0],NBDict,2)/(NBDict['evalation'][2]+len(wordsTest))
	p0*=NBDict['evalation'][0]/sum(NBDict['evalation'])
	p1*=NBDict['evalation'][1]/sum(NBDict['evalation'])
	p2*=NBDict['evalation'][2]/sum(NBDict['evalation'])
	p=[p0,p1,p2]
	return p.index(max(p))

def countNum(wordsList,trainSet):
	#print(wordsList,trainSet)
	NBDict={'evalation':[0,0,0]}
	for ix in trainSet.index:
		flag = trainSet.ix[ix,'pos_se_tag']
		NBDict['evalation'][flag]+=1
		for words in wordsList[ix]:
			if(words[0] in NBDict):
				NBDict[words[0]][flag]+=1
			else:
				NBDict[words[0]] = [0,0,0]
				NBDict[words[0]][flag]=1
	#print(NBDict)
	return NBDict

def tagTrainSet(trainSet):
	trainSet['pos_se_tag'] = 0
	for ix in trainSet.index:
		if(trainSet.ix[ix,'pos_se'] == '好'):
			trainSet.ix[ix,'pos_se_tag'] = 0
		elif(trainSet.ix[ix,'pos_se'] == '中'):
			trainSet.ix[ix,'pos_se_tag'] = 1
		elif(trainSet.ix[ix,'pos_se'] == '差'):
			trainSet.ix[ix,'pos_se_tag'] = 2

#def cutParagraphToWords(pa):
def getF(trainSet):
	dataSet=trainSet['content']
	wordsList=[]		
	for paragraph in dataSet.values:
		wordsList.append(cutParagraphToWords(paragraph))
	return wordsList

def cutParagraphToWords(paragraph):
	#print('step 1 cut paragraph to words')
	stopStrFileName='stopStr.txt'
	stopPath='{0}\\dict\\{1}'.format(os.getcwd(),stopStrFileName)

	# with open(stopPath,'r') as stopStrFile:
	stopStrFile=open(stopPath,'r',encoding='utf-8')
	f=stopStrFile.read()
	stopWords=f.split(' ')

	wordsCutList=[]
	paragraphToWords=pseg.cut(paragraph.split('>').pop())
	
	for word,flag in paragraphToWords:
		if(word not in stopWords and word != ' ' and word != ''):	
			wordsCutList.append([word,flag])
	return wordsCutList

def main():
	dataFileName='data11.xlsx'
	os.chdir('E:/Python-Workspace/machinelearning/5_navi_bayes/Sentiment_analysis')
	dataFilePath='{0}/NaiveBayesData/{1}'.format(os.getcwd(),dataFileName)
	# dataFilePath='E:\\Python-Workspace\\machinelearning\\5_navi_bayes\\Sentiment_analysis\\NaiveBayesData\\data11.xlsx'
	# 读取数据
	trainSet=pd.read_excel(dataFilePath)
	# 构建词袋：词语，标签
	wordsList=getF(trainSet)
	# 给词语打标签，一个评论中提取出来的词汇对应的标签就是这句评语的评价标签
	tagTrainSet(trainSet)
	# 统计词袋中词语的词频=词语：[l,m,n],l:好评语下的词频；m:中评语下的词频；n:差评语下的词频
	NBDict=countNum(wordsList,trainSet)
	# 分类，也就是朴素贝叶斯的计算公式
	p=classify([['差', 'a'], ['连单', 'a']],NBDict)
	print('ae: %d' % p)

if __name__ == '__main__':
	main()

