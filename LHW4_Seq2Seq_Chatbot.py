
# coding: utf-8

# # LHW4_Seq2Seq Chatbot

Group_member = "You Zhang & Changxu Zhang"

import pandas as pd
import re
import numpy as np
import random
import torch
import itertools
# from torch.jit import script, trace
# import torch.nn as nn
# from torch import optim
# import torch.nn.functional as F

MAX_LENGTH = 16
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
MIN_COUNT = 4


# ## Data Pre-Processing

# ### Read Data

with open('movie_lines.tsv', encoding='utf-8', errors='ignore') as f:
	data = f.read().split('\n')

lines = []
# count = 0
for line in data[:-1]:
	splitlist = line.rstrip().split("\t")
#     count += 1
	if splitlist[0].startswith('"'):
		splitlist[0] = splitlist[0][1:]
		if splitlist[-1].endswith('"'):
			splitlist[-1] = splitlist[-1][:-1]
	assert len(splitlist) > 3
	if len(splitlist) == 4: 
		splitlist.append(" ")
	if len(splitlist) > 5:
		splitlist = splitlist[:4] + [" ".join(splitlist[4:])]
	lines.append(splitlist)


# ### Clean Data

movie_lines = pd.DataFrame()
movie_lines["my"] = pd.Series(lines)
movie_lines[['lineID','characterID', 'movieID', 'character name', 'text of the utterance']]    = pd.DataFrame(movie_lines.my.values.tolist(), index = movie_lines.index)
movie_lines.drop(columns="my", inplace = True)
assert all(movie_lines["lineID"].str.startswith('L'))
assert all(movie_lines["characterID"].str.startswith('u'))
assert all(movie_lines["movieID"].str.startswith('m'))
movie_lines["lineID"] = movie_lines["lineID"].apply(lambda x: int(x[1:]))
movie_lines["characterID"] = movie_lines["characterID"].apply(lambda x: int(x[1:]))
movie_lines["movieID"] = movie_lines["movieID"].apply(lambda x: int(x[1:]))

def clean_text(text):
	text = re.sub(r"what's", "that is", text)
	text = re.sub(r"where's", "where is", text)
	text = re.sub(r"how's", "how is", text)
	text = re.sub(r"i'm", "i am", text)
	text = re.sub(r"he's", "he is", text)
	text = re.sub(r"she's", "she is", text)
	text = re.sub(r"it's", "it is", text)
	text = re.sub(r"that's", "that is", text)
	text = re.sub(r"\'ll", " will", text)
	text = re.sub(r"\'ve", " have", text)
	text = re.sub(r"\'re", " are", text)
	text = re.sub(r"\'d", " would", text)
	text = re.sub(r"\'re", " are", text)
	text = re.sub(r"won't", "will not", text)
	text = re.sub(r"can't", "cannot", text)
	text = re.sub(r"n't", " not", text)
	text = re.sub(r"n'", "ng", text)
	text = re.sub(r"'bout", "about", text)
	text = re.sub(r"'til", "until", text)
	
	text = re.sub(r"\{.*?\) ", " ", text)
	text = re.sub(r"\{.*?\} ", " ", text)
	text = re.sub(r"</.*?>", " ", text)
	text = re.sub(r"<.*?>", " ", text)
	text = re.sub(r"<.*?<", " ", text)
	text = re.sub(r"&.*?;", " ", text)
	text = re.sub(r'[^\d]%', " ", text)
	
	try:
		while re.match(r"^[^a-zA-Z]+.*", text).group() == text:
			text = text[1:]
	except:
		pass
	text = (text.lower()
			.replace('*', '')
			.replace('`', '')
			.replace('+', '')
			.replace('|', '')
			.replace(']', ' ')
			.replace('[', '')
			.replace('<', '')
			.replace('>', '.')
			.replace('=', ' ')
			.replace('~', '')
			.replace('\^', '')
			.replace('--', ' ')
			.replace('    ', ' ')
			.replace('   ', ' ')
			.replace('.', ' .')
			.replace(':', ' :')
			.replace(';', ' ;')
			.replace('!', ' !')
			.replace('?', ' ?')
			.replace('  ', ' '))
	return text

movie_lines['text of the utterance'] = movie_lines['text of the utterance'].apply(clean_text)


# ### Sort Data

movie_lines.sort_values(by=['movieID', 'lineID'], inplace = True)
movie_lines = movie_lines.reset_index(drop=True)


# ### Load Q&A Data

questions = []
answers = []
movieQandA = movie_lines['text of the utterance'].str.split(" ").values
characterID = movie_lines['characterID'].values
movieID = movie_lines['movieID'].values

for i in range(len(movieQandA) - 1):
	if len(movieQandA[i]) < MAX_LENGTH         and len(movieQandA[i+1]) < MAX_LENGTH         and not characterID[i] == characterID[i+1]         and movieID[i] == movieID[i+1]:
		questions.append(movieQandA[i])
		answers.append(movieQandA[i+1])
		
print("number of Q&As:", len(questions))


questionstuple = [tuple(question) for question in questions]
answerstuple = [tuple(answer) for answer in answers]
# ### Count Data

word2count = {}

for question in set(questionstuple+answerstuple):
    for word in question:
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

print("number of all words:", len(word2count))


# ### Trim and Index Data

keep_words = []
for key, value in word2count.items():
	if value >= MIN_COUNT:
		keep_words.append(key)

print("number of keep words:", len(keep_words))

word2index = {}
index2word = {0:" ", SOS_token: "SOS", EOS_token: "EOS"}
num_of_words = len(index2word)

pairs = []
for i in range(len(questions)):
	keep1 = True
	keep2 = True
	for word1 in questions[i]:
		if not word1 in keep_words:
			keep1 = False
			break
	for word2 in answers[i]:
		if not word2 in keep_words:
			keep2 = False
			break
	if keep1 and keep2:
		pairs.append([" ".join(questions[i]), " ".join(answers[i])])

print("number of keep Q&As:", len(pairs))


for i in range(len(pairs) - 1):
	for word in pairs[i][0].split(" "):
		if word not in word2index:
			word2index[word] = num_of_words
			index2word[num_of_words] = word
			num_of_words += 1
	if not pairs[i][1] == pairs[i+1][0]:
		for word in pairs[i][1].split(" "):
			if word not in word2index:
				word2index[word] = num_of_words
				index2word[num_of_words] = word
				num_of_words += 1


# ### Prepare Data for Model
# Referrence: https://pytorch.org/tutorials/beginner/chatbot_tutorial.html

def indexesFromSentence(word2index, sentence):
	return [word2index[word] for word in sentence.split(" ")] + [EOS_token]

def zeroPadding(l, fillvalue=0):
	return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=0):
	m = []
	for i, seq in enumerate(l):
		m.append([])
		for token in seq:
			if token == 0:
				m[i].append(0)
			else:
				m[i].append(1)
	return m

# Returns padded input sequence tensor and lengths
def inputVar(l, word2index):
	indexes_batch = [indexesFromSentence(word2index, sentence) for sentence in l]
	lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
	padList = zeroPadding(indexes_batch)
	padVar = torch.LongTensor(padList)
	return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, word2index):
	indexes_batch = [indexesFromSentence(word2index, sentence) for sentence in l]
	max_target_len = max([len(indexes) for indexes in indexes_batch])
	padList = zeroPadding(indexes_batch)
	mask = binaryMatrix(padList)
	mask = torch.ByteTensor(mask)
	padVar = torch.LongTensor(padList)
	return padVar, mask, max_target_len

# Returns all items for a given batch of pairs
def batch2TrainData(word2index, pair_batch):
	pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
	input_batch, output_batch = [], []
	for pair in pair_batch:
		input_batch.append(pair[0])
		output_batch.append(pair[1])
	inp, lengths = inputVar(input_batch, word2index)
	output, mask, max_target_len = outputVar(output_batch, word2index)
	return inp, lengths, output, mask, max_target_len


# ## Model

from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

class Chat(nn.Module):
	def __init__(self,hidden_size,num_of_words,dropout=0.1,batch_size = 64):
		super(Chat,self).__init__()
		self.encode_layers = 2
		self.decode_layers = 2
		self.embedding = nn.Embedding(num_of_words, hidden_size)
		self.embedding_dropout = nn.Dropout(dropout)
		
		self.hidden_size = hidden_size
		self.output_size = num_of_words
		self.lstm_ecd = nn.GRU(input_size = hidden_size,
							hidden_size = hidden_size,
							num_layers = self.encode_layers,
							dropout = dropout,
							bidirectional=True
						   )
		self.lstm_dcd = nn.GRU(input_size = hidden_size,
							hidden_size = hidden_size,
							num_layers = self.decode_layers,
							dropout = dropout
						   )
		self.lstm_ecd.flatten_parameters()
		self.lstm_dcd.flatten_parameters()
		self.concat = nn.Linear(hidden_size*2,hidden_size)
		self.out = nn.Linear(hidden_size,num_of_words)
		self.batch_size = batch_size
		self.teaching = 1.0
		
	def attention(self,hidden,ecd_out):
		energy = torch.sum(hidden*ecd_out,dim=2).t()
		return F.softmax(energy,dim=1).unsqueeze(1)
	def encode(self,x,length,h = None):
		embed = self.embedding(x)
		packed = torch.nn.utils.rnn.pack_padded_sequence(embed,length)
		out,h = self.lstm_ecd(packed,h)
		out,_ = torch.nn.utils.rnn.pad_packed_sequence(out)
		out = out[:,:,:self.hidden_size]+out[:,:,self.hidden_size:]
		return out,h
	def decode(self,ipt,last_hidden,encoder_out):
		embed = self.embedding(ipt)
		out,hidden = self.lstm_dcd(embed,last_hidden)
		attn_weights = self.attention(out,encoder_out)
		context = attn_weights.bmm(encoder_out.transpose(0,1))
		out = out.squeeze(0)
		context = context.squeeze(1)
		concat_ipt = torch.cat((out,context),1)
		concat_out = torch.tanh(self.concat(concat_ipt))
		output = self.out(concat_out)
		output = F.softmax(output,dim=1)
		return output,hidden
	def eval_result(self,ipt_seq,length,max_length):
		encoder_out,encoder_hidden = self.encode(ipt_seq,length)
		decoder_hidden = encoder_hidden[:self.decode_layers]
		decoder_input = torch.ones(1,1,device=device,dtype=torch.long)*SOS_token
		all_tokens = torch.zeros([0],device=device,dtype=torch.long)
		all_scores = torch.zeros([0],device=device)
		for i in range(max_length):
			decoder_output,decoder_hidden = self.decode(decoder_input,decoder_hidden,encoder_out)
			decoder_scores,decoder_input = torch.max(decoder_output,dim=1)
			all_tokens = torch.cat((all_tokens,decoder_input),dim=0)
			all_scores = torch.cat((all_scores,decoder_scores),dim=0)
			decoder_input = torch.unsqueeze(decoder_input,0)
		return all_tokens,all_scores
	
	def forward(self,input_variable,length,target,mask,max_target_length):
		encoder_outputs,encoder_hidden = self.encode(input_variable,length)
		decoder_input = torch.LongTensor([[SOS_token for i in range(self.batch_size)]])
		decoder_input = decoder_input.to(device)
		decoder_hidden = encoder_hidden[:self.decode_layers]
		return_list = []
		for t in range(max_target_length):
			decoder_out,decoder_hidden = self.decode(decoder_input,decoder_hidden,encoder_outputs)
			decoder_input=target[t].view(1,-1)
			return_list.append(decoder_out)
		return return_list

def maskNLLLoss(inp, target, mask):
	nTotal = mask.sum()
	crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)))
	loss = crossEntropy.masked_select(mask).mean()
	loss = loss.to(device)
	return loss, nTotal.item()

def train(pairs,num_of_words):
	batch_size= 64
	n_iteration = 10000
	hidden_size = 500
	clip=50
	training_batches = [batch2TrainData(word2index, [random.choice(pairs) for _ in range(batch_size)])
					  for _ in range(n_iteration)]

	print('Initializing ...')
	start_iteration = 1
	print_loss = 0  
	chat_model = Chat(hidden_size,num_of_words).to(device)
	chat_model.train()
	optimizer = optim.Adam(chat_model.parameters(),lr=0.0001)
	print("Training...")
	loss_all = 0
	for iteration in range(start_iteration, n_iteration + 1):
		training_batch = training_batches[iteration - 1]
		input_variable, lengths, target_variable, mask, max_target_len = training_batch
		input_variable = input_variable.to(device)
		lengths = lengths.to(device)
		target_variable = target_variable.to(device)
		mask = mask.to(device)
		result_list = chat_model(input_variable, lengths, target_variable,mask,max_target_len)
		loss = 0
		nTotal= 0
		for t in range(max_target_len):
			mask_loss,nTotal = maskNLLLoss(result_list[t],target_variable[t],mask[t])
			loss+=mask_loss
		optimizer.zero_grad()
		loss.backward()
		_ = torch.nn.utils.clip_grad_norm_(chat_model.parameters(),clip)
		optimizer.step()
		print("Iteration: %d"%iteration,"LOSS:",loss/nTotal)
	return chat_model
		
# Referrence: https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
def evaluate(chat_model,index2word,word2index, sentence, max_length=MAX_LENGTH):
	indexes_batch = [indexesFromSentence(word2index, sentence)]
	lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
	input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
	input_batch = input_batch.to(device)
	lengths = lengths.to(device)
	tokens, scores = chat_model.eval_result(input_batch, lengths, max_length)
	decoded_words = [index2word[token.item()] for token in tokens]
	return decoded_words

def evaluateInput(chat_model, index2word,word2index):
	input_sentence = ''
	while(1):
		try:
			input_sentence = input('Human > ')
			if input_sentence == 'q' or input_sentence == 'quit': break
			input_sentence = clean_text(input_sentence)
			output_words = evaluate(chat_model,index2word,word2index, input_sentence)
			output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
			print('Bot >', ' '.join(output_words))
		except:
			print('Sorry, I don\'t know what you mean')

# Run training iterations
def train_the_model():
	print("Starting Training!")
	chat_model = train(pairs,num_of_words)
	evaluateInput(chat_model,index2word,word2index)
	torch.save(chat_model,'model.pth')

def test_the_model():
	chat_model = torch.load('model.pth')
	evaluateInput(chat_model,index2word,word2index)

if __name__ == '__main__':
	print("Enter 1 to train, Enter 2 to test")	
	arg = input()
	if arg =="1":
		train_the_model()
	elif arg=="2":
		test_the_model()
	else:
		print("Illegal enter")
# ## Referrence

# #### https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
# #### https://github.com/Conchylicultor/DeepQA/
# #### https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
# #### https://github.com/Currie32/Chatbot-from-Movie-Dialogue/

# def parse_options(argv):
