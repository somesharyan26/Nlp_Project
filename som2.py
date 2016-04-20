import pickle
from collections import Counter
import math
import os
#import re

plus = 'pos2.txt'
minus = 'neg2.txt'
pos,neg,pos_w,neg_w,voc,db,voc={},{},{},{},{},{},{}
tot_freq_pos=0
tot_freq_neg=0
voc_size=0
def unWanted_char(name):
	newWord=[]
	stop = open('stopwords.txt','r')
	for word in stop:
		newWord.append(word[ :-1])	#remove \n in list
	inFile = open(name,'r')
	data = inFile.read()
	inFile.close()
	for stopWord in newWord:
		data = data.replace(" "+stopWord+" "," ")
		data = data.replace("\n"+stopWord+" ","\n")
		data = data.replace(" "+stopWord+"\n","\n")
	data = data.lower()
	data = data.replace("\"", "")
	data = data.replace(",", "")
	data = data.replace(".", " ")
	data = data.replace("!", " ")
	data = data.replace("/", " ")
	data = data.replace("-", " ")
	data = data.replace("?", " ")
	data = data.replace("`", " ")
	data = data.replace("\'", " ")
	data = data.replace(";", " ")
	data = data.replace("&", " ")
	data = data.replace("\\", " ")
	data = data.replace("*", " ")
	data = data.replace("$", "")
	data = data.replace(":", " ")
	data = data.replace(" s ", " ")
	data = data.replace("\ns ", "")
	data = data.replace(" d "," ")

	s='0123456789'
	for i in s:
		data = data.replace(i,"")	
	outFile = open('out_'+name,'w')
	outFile.write(data)
	outFile.close()

def destroy(fname):
	if os.path.exists(fname):
		os.remove(fname)

def divide10(part):
	infile = open(part,'r')
	for i in range(1,11):
		destroy(str(i)+part)
		outfile = open(str(i)+part,'w')
		for j in range(50):
			outfile.write(infile.readline())
		outfile.close()
	infile.close()

def total(db1):
	c=0
	for word in db1:
		c += db1[word]
	return c

def pickleOut(temp,fname):
	pickle_out = open(fname+'.p', 'wb')
	pickle.dump(temp, pickle_out)
	pickle_out.close()
	
def pickleIn(fname):
	pickle_in = open(fname+'.p','rb')
	db1 = pickle.load(pickle_in)
	pickle_in.close()
	return db1

def lis2dic(l):
	counts = Counter(l)
	#counts = {i:counts[i] for i in counts if counts[i]>=2}
	return counts

def tokenize(fname):
	infile = open(fname,'r')
	l=[]
	for line in infile.readlines():
		ls = line.split()
		ls = [i.strip('\n') for i in ls if i!='']
		ls += ['*','**']
		l += ls
	return l

def get_bigrams(fname):
	infile = open(fname,'r')
	bi_grams=[]
	for line in infile.readlines():
		ls = line.split()
		ls = [i.strip('\n') for i in ls if i != '']
		bi=[]
		if ls[0]:
			bi=[('*',ls[0])]
		n = len(ls)
		for i in range(n-1):
			bi.append((ls[i],ls[i+1]))
		if ls:
			bi.append((ls[n-1],'**'))
		bi_grams += bi		
	return bi_grams
def str_to_bigrams(doc):
	ls = doc.split()
	ls = [i.strip('\n') for i in ls if i != '']
	bi_grams=[]
	if ls:
		bi_grams=[('*',ls[0])]
	n = len(ls)
	for i in range(n-1):
		bi_grams.append((ls[i],ls[i+1]))
	if ls:
		bi_grams.append((ls[n-1],'**'))
	return bi_grams
	
def p(bigram,flag):
	global pos,pos_w,neg_w,neg,voc,db,tot_freq_pos,tot_freq_neg,voc_size,all_bigrams
	db1 = neg
	db2 = neg_w
	if flag==1:
		db1 = pos
		db2 = pos_w
	tot = len(db1)
	if bigram in db1:
		if bigram[0] in db2:
			return math.log(db1[bigram]+1)-math.log(db2[bigram[0]]+voc_size)
		else:
			return math.log(db1[bigram]+1)-math.log(1+voc_size)
	else: 
		return math.log(1)-math.log(tot+voc_size)

def  create_model():
	global pos,pos_w,neg_w,neg,voc,db,tot_freq_pos,tot_freq_neg,voc_size,all_bigrams
	db = {}
	for bigram in all_bigrams:
		db[bigram]=[p(bigram,0),p(bigram,1)]
	destroy('MNBmodel.p')
	pickleOut(db,'MNBmodel')

def q(doc,clas):
	global pos,pos_w,neg_w,neg,voc,db,tot_freq_pos,tot_freq_neg,voc_size
	res = 0
	doc = str_to_bigrams(doc)
	theta = len(neg)
	if clas == 1:
		theta = len(pos)
	for bigram in doc:
		if bigram in all_bigrams:
			res += db[bigram][clas]
		else:
			res += math.log(1)-math.log(theta+voc_size)
	return res

def predict(doc):
	if q(doc,1)>=q(doc,0):
		return 1 #pos
	else:
		return 0 #neg
	
def result():
	global pos,pos_w,neg_w,neg,voc,db,tot_freq_pos,tot_freq_neg,voc_size,all_bigrams
	sum_acc =0
	for i in range(1,11):
		pos,neg,voc,db,all_bigrams={},{},{},{},{}
		tot_freq_pos=0
		tot_freq_neg=0
		voc_size=0
		traindat1,traindat2 = '',''
		for j in range(1,11):
			
			if j != i:
				inplus = open(str(j)+plus)	
				traindat1 += inplus.read()
				inplus.close()
				inminus = open(str(j)+minus)
				traindat2 += inminus.read()
				inminus.close()
		destroy('train_'+plus)
		intrain = open('train_'+plus,'w')
		intrain.write(traindat1)
		intrain.close()
		destroy('train_'+minus)
		intrain = open('train_'+minus,'w')
		intrain.write(traindat2)
		intrain.close()

		ls =  (traindat1 + traindat2).split('\n')
		print 'length of taining dat :',len(ls)
		
		pbfl =get_bigrams('train_'+plus)
		pos = lis2dic(pbfl)
		pf = tokenize('train_'+plus)
		pos_w = lis2dic(pf)
		
		nbfl = get_bigrams('train_'+minus)
		neg = lis2dic(nbfl)
		nf = tokenize('train_'+minus)
		neg_w = lis2dic(nf)
		
		all_bigrams = lis2dic(pbfl+nbfl)
		voc = lis2dic(pf+nf)
		voc_size = len(voc)+2
		tot_freq_pos = total(pos_w)
		tot_freq_neg = total(neg_w)
		create_model()
		db = pickleIn('MNBmodel')
		fp,fn,tp,tn = 0,0,0,0
		tes = open(str(i)+plus,'r').read().split('\n')
		for doc in tes:
			if predict(doc)==1:
				tp += 1
			else:
				fn += 1
		testl = open(str(i)+minus,'r').readlines()
		for doc in testl:
			if predict(doc)==0:
				tn += 1
			else:
				fp += 1

		print fp,fn,tp,tn 
		accuracy = (tp+tn)/float(tp+tn+fn+fp)
		print 'accuracy',accuracy
		sum_acc  +=  accuracy
	avg_acc = sum_acc/10.0
	print avg_acc

unWanted_char('pos2.txt')
unWanted_char('neg2.txt')
divide10('out'+plus)
divide10('out'+minus)
result()	