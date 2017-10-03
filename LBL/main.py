from __future__ import print_function

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import re


trainFile = "train.txt"
gloveModel = "glove.6B.50d.txt"


def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding

    print ("Done.",len(model)," words loaded!")
    return  model

def checkWordsInTrain(trainFile, dictionary):
    """ return list of words in training data that are not in dictionary"""
    f = open(trainFile, 'r')
    notfound = []
    for line in f:
        for word in line.split():
            if word not in dictionary:
                notfound.append(word)
    return notfound    

def prepareTrainData(trainFile):
    """1. lowercase everything
       2. add space before punctuation"""
    inputFile = open(trainFile, 'r')
    content = inputFile.read()
    inputFile.close()
    with open(trainFile, 'w') as outputFile:
        # 1.
        lowercase = content.lower()
        # 2.
        paddpunct = re.sub('(?<! )(?=[.,;:!?()-''])|(?<=[.,;:!?()-])(?! )', r' ', lowercase)
        outputFile.write(paddpunct)
    outputFile.close()
    return

def deleteNewWords(trainFile, dictionary):
    """ delete words not in dictionary"""
    delete_list = checkWordsInTrain(trainFile, dictionary)
    inputFile = open(trainFile, 'r')
    lines = inputFile.readlines()
    inputFile.close()
    outputFile = open(trainFile, 'w')
    for line in lines:
        for word in delete_list:
            line = line.replace(word, "")
        outputFile.write(line)
    outputFile.close()
    
def getVocabSize(trainFile):
    f = open(trainFile, 'r')
    d = {}
    for line in f:
        for word in line.split():
            d[word] = 1
    return len(d)
            

# Load vector representation of words (GLOVE pretrained)
glove = loadGloveModel(gloveModel)
# dictionary: mapping words to vectors

# Clean training data
prepareTrainData(trainFile)
deleteNewWords(trainFile, glove)
assert(0 == len(checkWordsInTrain(trainFile, glove)))

print(getVocabSize(trainFile))

def makeVecRepresentationMatrix(trainFile, dictionary):
    """ Construct a tensor matrix with a vector representation of the words
    in the vocab"""
    R = torch.FloatTensor(getVocabSize(trainFile), 50)
    f = open(trainFile, 'r')
    d = {}
    w2i, i2w = {}, {}
    i = 0
    for line in f:
        for word in line.split():
            if word not in d:
                d[word] = 1
                R[i] = torch.FloatTensor(dictionary[word])
                w2i[word] = i
                i2w[i] = word
                i += 1
    return R, w2i, i2w

R, w2i, i2w = makeVecRepresentationMatrix(trainFile, glove)

VOCAB_SIZE = getVocabSize(trainFile)
HIDDEN_LAYER_SIZE = 50
CONTEXT_SIZE = 5

class LBL(nn.Module):  

    def __init__(self, vocab_size, hid_layer_size, context_size, R):
        super(LBL, self).__init__()
        # init configuration
        self.vocab_size = vocab_size
        self.hid_layer_size = hid_layer_size
        # embedding layers
        self.word_embeds = nn.Embedding(vocab_size, hid_layer_size)
        # Weight matrix, d to hidden layer
        self.C = nn.Linear(hid_layer_size * context_size, hid_layer_size, bias=False)
        # Bias in softmax layer
        self.bias = nn.Parameter(torch.ones(vocab_size))

        self.init_weight(R)
    
    def get_train_parameters(self):
        params = []
        for param in self.parameters():
            if param.requires_grad == True:
                params.append(param)
        return params

    def init_weight(self, glove_weight):
    	assert(glove_weight.size() == (self.vocab_size, self.hid_layer_size))
        self.word_embeds.weight.data.copy_(glove_weight)
        self.word_embeds.weight.requires_grad = False

    def forward(self, context_vect):      
        return F.log_softmax(pytorch.mm(R, self.C(context_vect)) + self.bias)


def get_word_index(word, w2i):
    if word not in w2i: 
        return w2i["unk"]
    else:
        return w2i[word]

def make_context_vector(wordlist, w2i): # TODO: change this

    embeddinglist = []
    for word in wordlist:

        embeddinglist.append(get_word_index(word, w2i))
    vec = torch.LongTensor(embeddinglist)
    # 
    return vec #vec.view(1,-1)

def make_target(word, w2i):  # TODO:change this
    return torch.LongTensor([get_word_index(word, w2i)])

def print_params(model):
    for param in model.parameters():
        print(param)

def test(testFile, model):
    """Return - log likelihood of the training data set base on the model"""
    print("not implemented")

        
def train(R, trainFile, w2i, epochs=30, lr=0.01):
    """Train model with trainFile"""
    model = LBL(VOCAB_SIZE, HIDDEN_LAYER_SIZE, CONTEXT_SIZE, R)

    loss_function = nn.NLLLoss() 
    optimizer = optim.SGD(model.get_train_parameters(), lr = 0.01)

    for epoch in range(30):
        f = open(trainFile, 'r')
        text = f.read()
        word_list = []
        for word in text:
            # Continue until we see at least conext_size words
            if len(word_list) < CONTEXT_SIZE:
                word_list.append(word)
                continue
            
            # Step 1. clear out gradients
            model.zero_grad()
            
            # Step 2. Get intput and target
            context_vect = autograd.Variable(make_context_vector(word_list, w2i))
            target = autograd.Variable(make_target(word, w2i))
            
            # Step 3. Run forward pass
            log_probs = model(context_vect)
            
            # Step 4. Compute loss, gradients and update parameters
            loss = loss_function(log_probs, target)
            loss.backward()
            optimizer.step()
            
            # Update the context vector
            word_list.pop(0)
            word_list.append(word)

    return model 

print("training")
train(R, trainFile, w2i)