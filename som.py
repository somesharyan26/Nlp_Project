import os
import pickle
from collections import Counter
import math

def unWanted_char(name):
	newWord=[]
	stop = open('stopwords.txt','r')
	for word in stop:
		newWord.append(word[ :-1])	
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

	s='0123456789'
	for i in s:
		data = data.replace(i,"")	
	outFile = open('out'+name,'w')
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

def classifier(doc,clas):
	global pos,neg,voc,db,tot_freq_pos,tot_freq_neg,voc_size
	res = 0
	doc = doc.split()
	theta = tot_freq_neg
	if clas == 1:
		theta = tot_freq_pos
	for word in doc:
		if word in voc:
			res += db[word][clas]
		else:
			res += math.log(1)-math.log(theta+voc_size)
	return res

def prediction(doc):
	if classifier(doc,1)>=classifier(doc,0):
		return 1 
	else:
		return 0 

def probability(word,flag):
	global pos,neg,voc,db,tot_freq_pos,tot_freq_neg,voc_size
	db1 = neg
	tot = tot_freq_neg
	if flag==1:
		db1 = pos
		tot = tot_freq_pos
	if word in db1:
		return math.log(db1[word])-math.log(tot+voc_size)
	else: 
		return math.log(1)-math.log(tot+voc_size)

def  model():
	global pos,neg,voc,db,tot_freq_pos,tot_freq_neg,voc_size
	db = {}
	for word in voc:
		db[word]=[probability(word,0),probability(word,1)]
	pickleOut(db,'MNBmodel')
	
def total(db1):
	c=0
	for word in db1:
		c += db1[word]
	return c

def pre_patch():
	global pos,neg,voc,db,tot_freq_pos,tot_freq_neg,voc_size
	voc_size = len(voc)
	tot_freq_pos = total(pos)
	tot_freq_neg = total(neg)
	ls = [voc_size,tot_freq_neg,tot_freq_pos]
	pickleOut(ls,'net_word_freq')

def pickleOut(temp,fname):
	pickle_out = open(fname+'.p', 'wb')
	pickle.dump(temp, pickle_out)
	pickle_out.close()

def pickleIn(fname):
	pickle_in = open(fname+'.p','rb')
	db1 = pickle.load(pickle_in)
	pickle_in.close()
	return db1

def list_to_dict(l):
	counts = Counter(l)
	counts = {i:counts[i] for i in counts if counts[i]>=2}
	return counts

def token(fname):
	infile = open(fname,'r')
	l=[]
	for line in infile.readlines():
		ls = line.split()
		ls = [i.strip('\n') for i in ls if i!='']
		l += ls
	return l

def frequency():
	pfl =token('train_pos2.txt')
	temp = list_to_dict(pfl)
	pickleOut(temp,'pos_word_freq')

	nfl = token('train_neg2.txt')
	temp = list_to_dict(nfl)
	pickleOut(temp,'neg_word_freq')
	
	temp = list_to_dict(pfl+nfl)
	pickleOut(temp,'all_word_freq')
	
def result():
	global pos,neg,voc,db,tot_pos_freq,tot_neg_freq,voc_size
	sum_acc =0
	for i in range(1,11):
		pos={}
		neg={}
		voc={}
		db={}
		tot_pos_freq=0
		tot_neg_freq=0
		voc_size=0
		traindat1,traindat2 = '',''
		for j in range(1,11):
			if j != i:
				testpos = open(str(j)+'pos2.txt')	
				traindat1 += testpos.read()
				testpos.close()
				testneg = open(str(j)+'neg2.txt')
				traindat2 += testneg.read()
				testneg.close()
		destroy('train_pos2.txt')
		intrain = open('train_pos2.txt','w')
		intrain.write(traindat1)
		intrain.close()
		destroy('train_neg2.txt')
		intrain = open('train_neg2.txt','w')
		intrain.write(traindat2)
		intrain.close()

		ls =  (traindat1 + traindat2).split('\n')
		print 'length of taining dat :',len(ls)
		frequency()
		pos = pickleIn('pos_word_freq')
		neg = pickleIn('neg_word_freq')
		voc = pickleIn('all_word_freq')
		pre_patch()
		ls =  pickleIn('net_word_freq')
		voc_size = ls[0]
		tot_freq_neg = ls[1]
		tot_freq_pos = ls[2]

		model()
		db = pickleIn('MNBmodel')
		fp,fn,tp,tn = 0,0,0,0
		tes = open(str(i)+'pos2.txt','r').read().split('\n')
		for doc in tes:
			if prediction(doc)==1:
				tp += 1
			else:
				fn += 1
		testl = open(str(i)+'neg2.txt','r').readlines()
		for doc in testl:
			if prediction(doc)==0:
				tn += 1
			else:
				fp += 1 
		accuracy = (tp+tn)/float(tp+tn+fn+fp)
		print 'accuracy',accuracy
		sum_acc  +=  accuracy
	avg_acc = sum_acc/10.0
	print avg_acc

unWanted_char('pos2.txt')
unWanted_char('neg2.txt')
divide10('pos2.txt')
divide10('neg2.txt')
result()