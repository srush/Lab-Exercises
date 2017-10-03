from __future__ import print_function

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import re

HIDDEN_LAYER_SIZE = 50
CONTEXT_SIZE = 5
VEC_SIZE = 50

trainFile = "train.txt"
testFile = "test.txt"
gloveModel = "glove.6B.50d.txt"


def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    # iterate through every word-vec representation
    for line in f:
        # split and lower case inputs
        splitLine = line.split()
        word = splitLine[0].lower()
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model

def checkWordsInTrain(trainFile, dictionary):
    """return list of words in training data that are not in dictionary"""
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
        # clean training data
        lowercase = content.lower()
        paddpunct = re.sub('(?<! )(?=[.,;:!?()-''])|(?<=[.,;:!?()-])(?! )', r' ', lowercase)
        outputFile.write(paddpunct)
    outputFile.close()
    return

def deleteNewWords(trainFile, dictionary):
    """delete words not in dictionary"""
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
    d = {}
    with open(trainFile, 'r') as f:
        for line in f:
            for word in line.split():
                d[word] = 1
    return len(d)

def makeVecRepresentationMatrix(trainFile, word2vec, vocab_size):
    """Construct a tensor matrix with a vector representation of the words
    in the vocab"""
    unknown = "unk"
    R = torch.FloatTensor(vocab_size, VEC_SIZE)
    with open(trainFile, 'r') as f:
        # lookup table
        d = {}
        w2i, i2w = {}, {}
        i = 0
        for line in f:
            for word in line.split():
                if word not in d:
                    if word not in word2vec:
                        continue
                    d[word] = 1
                    R[i] = torch.FloatTensor(word2vec[word])
                    w2i[word] = i
                    i2w[i] = word
                    i += 1
        R[i] = torch.FloatTensor(word2vec["unk"])
        w2i["unk"] = i
        i2w[i] = "unk"
    return R, w2i, i2w


class LBL(nn.Module):

    def __init__(self, vocab_size, hid_layer_size, context_size, R):
        super(LBL, self).__init__()
        # init configuration
        self.vocab_size = vocab_size
        self.context_size = context_size
        print("vocab_size=", vocab_size)
        print("R.shape=", R.size())
        self.hid_layer_size = hid_layer_size
        # embedding layers
        self.word_embeds = nn.Embedding(vocab_size, hid_layer_size)
        # Weight matrix, d to hidden layer
        self.C = nn.Linear(hid_layer_size * context_size, hid_layer_size, bias=False)
        # Bias in softmax layer
        self.bias = nn.Parameter(torch.ones(vocab_size)).view(self.vocab_size, 1)

        self.init_weight(R)
        self.R = autograd.Variable(R)

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
        context_vect = self.word_embeds(context_vect)
        context_vect = context_vect.view(1, self.context_size * self.hid_layer_size)
        model_vect = self.C(context_vect).view(self.hid_layer_size, 1)
        final_vect = torch.mm(self.R, model_vect) + self.bias
        final_vect = F.log_softmax(final_vect).view(1, self.vocab_size)
        return final_vect

def make_context_vector(wordlist, w2i):
    unknown = "unk"
    embeddinglist = []
    for word in wordlist:
        if word not in w2i:
            embeddinglist.append(w2i[unknown])
        else:
            embeddinglist.append(w2i[word])
    vec = torch.LongTensor(embeddinglist)
    return vec #vec.view(1,-1)

def make_target(word, w2i):
    unknown = "unk"
    if word in w2i:
        return torch.LongTensor([w2i[word]])
    else:
        return torch.LongTensor([w2i[unknown]])

def print_params(model):
    for param in model.parameters():
        print(param)

def test(testFile, model):
    """Return - log likelihood of the training data set base on the model"""
    loss_function = nn.NLLLoss()
    tot_loss = 0
    model.eval()

    f = open(trainFile, 'r')
    text = f.read()
    word_list = []
    for word in text:
        # Continue until we see at least conext_size words
        if len(word_list) < CONTEXT_SIZE:
            word_list.append(word)
            continue

        # Step 1. Get intput and target
        context_vect = autograd.Variable(make_context_vector(word_list, w2i))
        target = autograd.Variable(make_target(word, w2i)).view(1)

        # Step 2. Run forward pass
        log_probs = model(context_vect)

        # Step 3. Compute loss
        loss = loss_function(log_probs, target)
        tot_loss += loss.data.numpy()[0]

        # Update the context vector
        word_list.pop(0)
        word_list.append(word)

    f.close()
    return tot_loss


def train(R, trainFile, w2i, epochs=30, lr=0.01):
    """Train model with trainFile"""
    model = LBL(VOCAB_SIZE, HIDDEN_LAYER_SIZE, CONTEXT_SIZE, R)
    model.train()
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.get_train_parameters(), lr = 0.01)

    for epoch in range(30):
        with open(trainFile, 'r') as f:
            text = f.read()
            word_list = []
            i = 0
            total_loss = 0
            for word in text:
                i += 1
                if i % 1000 == 0:
                    print("epoch=%d, word=%d/%d\n" % (epoch, i, len(text)))
                # Continue until we see at least conext_size words
                if len(word_list) < CONTEXT_SIZE:
                    word_list.append(word)
                    continue

                # Step 1. clear out gradients
                optimizer.zero_grad()

                # Step 2. Get intput and target
                context_vect = autograd.Variable(make_context_vector(word_list, w2i))
                target = autograd.Variable(make_target(word, w2i)).view(1)

                # Step 3. Run forward pass
                log_probs = model(context_vect)

                # Step 4. Compute loss, gradients and update parameters
                loss = loss_function(log_probs, target)
                print("loss intermediate=", loss)
                total_loss += loss.data.numpy()[0]
                loss.backward()
                optimizer.step()

                # Update the context vector
                word_list.pop(0)
                word_list.append(word)
        print(total_loss)

    return model


# Load vector representation of words (GLOVE pretrained)
# a dictionary mapping words to vectors
word2vec = loadGloveModel(gloveModel)

# Clean training data
prepareTrainData(trainFile)

# vocab size including unknown
VOCAB_SIZE = getVocabSize(trainFile) + 1 # +1 unk
print (VOCAB_SIZE)
R, w2i, i2w = makeVecRepresentationMatrix(trainFile, word2vec, VOCAB_SIZE)

print("training...")
model = train(R, trainFile, w2i)
test(testFile, model)

