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
    return model

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
    i = 0
    for line in f:
        for word in line.split():
            if word not in d:
                d[word] = 1
                R[i] = torch.FloatTensor(dictionary[word])
                i += 1
    return R

R = makeVecRepresentationMatrix(trainFile, glove)

VOCAB_SIZE = getVocabSize(trainFile)
HIDDEN_LAYER_SIZE = 50
CONTEXT_SIZE = 5

class LBL(nn.Module):  

    def __init__(self, vocab_size, hid_layer_size, context_size, R):
        super(LBL, self).__init__()
        # Weight matrix, inputs to hidden layer
        self.C = nn.Linear(hid_layer_size * context_size, hid_layer_size, bias=False)
        # Bias in softmax layer
        self.bias = nn.Parameter(torch.ones(vocab_size))

    def forward(self, context_vect):      
        return F.log_softmax(pytorch.mm(R, self.C(context_vect)) + self.bias)

def make_context_vector(wordlist, dictionary):
    embeddinglist = []
    for word in wordlist:
        embeddinglist += dictionary[word]
    vec = torch.FloatTensor(embeddinglist)
    # 
    return vec #vec.view(1,-1)

def make_target(word, dictionary):
    return torch.FlotTensor(dictionary[word])

def print_params(model):
    for param in model.parameters():
        print(param)
model = LBL(VOCAB_SIZE, HIDDEN_LAYER_SIZE, CONTEXT_SIZE, R)


loss_function = nn.NLLLoss() 
optimizer = optim.SGD(model.parameters(), lr = 0.01)



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
        context_vect = autograd.Variable(make_context_vector(word_list, glove))
        target = autograd.Variable(make_target(word))
        
        # Step 3. Run forward pass
        log_probs = model(context_vect)
        
        # Step 4. Compute loss, gradients and update parameters
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()
        
        # Update the context vector
        word_list.pop(0)
        word_list.append(word)
        