import pandas as pd
import numpy as np 
import jieba
import collections
import re 
import os
import json

class ReadData(object):
	"""
		Args：
			sentence_lenth：指的是评论的长度（根据分词后的词的个数）
			mode：指的是运用模式，‘train’适用于训练神经网络的时候，此时保存词典等信息
			batch_size: 这里的batch要适配于下面的nextbatch()方法中的batch
			vocabulary_size：指的是构造的词典的词数
			load_size:指的是需要加载的语料的数量。
		"""
	def __init__(self, sentence_lenth, mode ='train', batch_size=50, vocabulary_size = 2000, load_size= 1840):
		self.stopword = "，。！ \n*的了是也"
		self.mode = mode
		self.offset = 0
		self.epochs = 0
		self.load_size = load_size
		self.vocabulary_size = vocabulary_size
		self.sentence_lenth = sentence_lenth
		self.file = "./log/word_dict.json"
		self.batchsize = batch_size

	def Test(self, text):
		textword = [jieba.cut(text)]
		# print(textword)
		textword = self.normalize(textword, self.stopword)
		if os.path.exists(self.file):
			# with open(self.file,'rb') as line:
			for line in open(self.file,'rb'):
				line.decode()     ## 在写的时候是用utf-8的字节存储的
				word_dict = json.loads(line)
				# word_dict, word_list = self.make_dictionary(textword, self.vocabulary_size)
			word_list = word_dict.keys()   		## 获取的是键值list
			# print(word_list)
			textnum = self.text2num(textword, word_dict, word_list, self.sentence_lenth)
			zero = np.zeros([self.batchsize-1, self.sentence_lenth], dtype=np.int32)
			textnum = np.append(textnum, zero, axis=0)
			textnum = np.int32(textnum)
			return textnum
		else:
			print("字典不存在，请初始化后再应用此方法")
			return 0

	def Train_Init(self):
		"""
			在训练之前的对语料进行统一的初始化，包括分词和进行词典、词向量的构造等。
		"""
		data, self.label = self.Readcsv(self.load_size)
		textword = self.text2word(text =data , mode = False)
		textword = self.normalize(textword, self.stopword)
		word_dict, self.word_list = self.make_dictionary(textword, self.vocabulary_size)
		self.textnum = self.text2num(textword, word_dict, self.word_list, self.sentence_lenth)
		
		num = np.arange(self.load_size)		
		np.random.shuffle(num)
		self.textnum = self.textnum[num]
		# print(self.textnum.shape)
		self.label = self.label[num]
		# print(type(self.textnum))
		


	def Readcsv(self, load_size):
		"""
		对语料的读取和相应属性的读取
		Args：
			load_size:指的是要加载的数据量，最大值为4985条
		"""
		Assessdic = {'好评':0, '中评':1, '差评':2 }
		if self.mode == 'train':
			data_average = pd.read_csv(".\\Data\\JD_Crawler_Average.csv",encoding='gbk')
			data_bad = pd.read_csv(".\\Data\\JD_Crawler_Bad.csv",encoding='gbk')
			data_good = pd.read_csv(".\\Data\\JD_Crawler_Now.csv",encoding='gbk')
			data_all = data_average.append(data_bad)
			data_all = data_all.append(data_good)
			for filename in os.listdir('.\\Data'):
				if '2' not in filename:
					continue
				if 'csv' not in filename:
					continue
				path = os.path.join('.\\Data', filename)
				data = pd.read_csv(path, encoding='gbk')
				data_all = data_all.append(data)
			# print(len(data_all))
			data_label = data_all['Assess'].values
			label = [Assessdic.get(Assess) for Assess in data_label]
			label = label[0:load_size]
			# print(len(label))
			label = np.array(label)
			data = list(data_all['Comment'].values)   ### 这里返回的是句子组成的list
			data = data[0:load_size]
			# print(type(data))
		else:
			data_all = pd.read_csv(".\\Data\\JD_Crawler_Former.csv", encoding = 'gbk')
			# print(len(data_all))
			data_label = data_all['Assess'].values
			label = [Assessdic.get(Assess) for Assess in data_label]
			# print(len(label))
			self.load_size = len(data_label)
			label = np.array(label)
			data = list(data_all['Comment'].values)   ### 这里返回的是句子组成的list
		return data, label

	def text2word(self, text, mode):
		"""
		text：输入文本，要求格式是2维list，[batch, sentence]
		mode:True是用关键词提取；
			False是直接分词提取
		"""
		## 提取关键词作为词典
		if mode == True:
			words = [jieba.analyse.extract_tags(sentence) for sentence in text]   ## extract_tags()直接返回list
		else:		## 以直接分词结果作为词典
			words = [list(jieba.cut(sentence)) for sentence in text]          ## jieba 的cut方法返回的是一个迭代器
		return words


	def normalize(self, words, stopword):
		"""
		过滤停用词，标点等
		"""
		## 去掉666等数字
		words = [[word for word in sentence if not re.match(r'\d{1,4}', word)]for sentence in words]
		words = [[word for word in sentence if word not in "0123456789"]for sentence in words]
		word_nor = [[word for word in sentence if word not in stopword] for sentence in words]
		return word_nor

	def make_dictionary(self, words, vocabulary_size):
		"""
		Args：
			words：每行为句子，列为分词的矩阵
			vocabulary——size：统计分词的个数
		构造规则词典规则：以出现频数多到少排序，取前vocabulary_size构造词典，同时不再词典内的就以NaN代替。
		"""
		wordlist = [word for sentence in words for word in sentence] # 展开成一维，统计词
		word_dict = collections.Counter(wordlist).most_common(vocabulary_size-1) ## 按频数排序，返回一个个tuple（“word”，频数）
		word_list = [word[0] for word in word_dict]  
		word_list.insert(0, 'NaN')   ## 插入无效词
	    # print(word_list)
	    # print(type(word_list))
		word_dict = {word:index for index,word in enumerate(word_list)} ## enumerate()枚举方式返回序号和值
		## 保存词典数据，保证下次的编码还是同样的编码规则
		if os.path.exists(self.file):
			with open(self.file,'wb') as file:
				data = json.dumps(word_dict).encode()
				# print(data)
				file.write(data)
		else:
			os.mkdir('./log')
			with open(self.file,'wb') as file:
				data = json.dumps(word_dict).encode()
				# print(data)
				file.write(data)
		return word_dict, word_list

	def text2num(self, words, word_dict, word_list, sentence_lenth):
		"""
			Args:
			words: 已经分好次的数据，格式[batch, words]
			word_dict: 已经统计好的词典（根据词频排序）
			word_list: 词典中统计的词
			sentenc_lenth: 用于规则化的句子长度（补0 或者 删除）
			
			Returns:
			textnum:未经过规则化的词表（长短不一）
			textnum_nor:经过规则化（补0，删除）的词表 site：[batch, sentence_lenth]
		"""
		# if len(words)<2:
		# 	textnum = [[word_dict.get(word, 0) for word in sentence if word in word_list]for sentence in words]
		# else:
		# 	textnum = [word_dict.get(word, 0) for word in words]
		## 把文本数字化
		textnum = [[word_dict.get(word, 0) for word in sentence if word in word_list]for sentence in words]
		# print(words)
		output = []
		## 把文本规则化， 多的裁剪，少的补零
		for sentence in textnum:
			if len(sentence)>=sentence_lenth:
				word = sentence[0:sentence_lenth]
			else:
				zeros = [0]*(sentence_lenth-len(sentence))
				word = sentence + zeros
			output.append(word)
		output = np.array(output)
		# print(output.shape)
		return output

	def NextBatch(self, batchsize):
		"""
		抽取下一个批次的数据，返回的是array：[batch, sentence_lenth]
		Args:
			batchsize:指的是下次返回的批次的数量
		"""
		start = self.offset
		self.offset +=batchsize
		if self.offset >self.load_size:
			self.epochs += 1 
			print("finish training "+str(self.epochs)+" times")
			num = np.arange(self.load_size)
			np.random.shuffle(num)       	## 随机打乱num的顺序
			self.textnum = self.textnum[num] ## 按照num 的顺序来重新排序
			self.label = self.label[num]
			start = 0
			self.offset = batchsize
		end = self.offset
		return self.textnum[start:end], self.label[start:end]


if __name__ == '__main__':
	test = ReadData(20, mode='train', load_size=100)
	test.Train_Init()
	for i in range(0,20):
		a,b = test.NextBatch(20)

	# # a, b = test.NextBatch(20)
	# print(a.shape)
	# print(a[0])
	# print(b[0])

	# test = ReadData(20, mode='train', load_size=100)
	# num = test.Test("给二宝买的奶粉，真的很棒，性价比高，宝宝爱喝，我也安心")
	# print(num.shape)
	# print(num)