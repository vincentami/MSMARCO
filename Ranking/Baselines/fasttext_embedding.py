# -*-coding:UTF-8 -*-
#!/usr/bin/python
'''
Created on  20180427

@author: vincent
'''
import sys
import numpy as np
import fasttext

def fasttext_embedding(sammpleFile, modelFile, vecFile):

    wordMap = {}

    # skmodel = fasttext.train_unsupervised(sammpleFile, model='skipgram', epoch=6, lr=0.1, dim = 128, thread=16, wordNgrams = 2, loss = 'hs')
    skmodel = fasttext.train_unsupervised(sammpleFile, model='skipgram', epoch=6, lr=0.1, dim = 128, thread=16, wordNgrams = 2, loss = 'ns')

    # for x in ["dogo", "dogo meño"]:
    #     testVec = skmodel.get_word_vector(x)
    #     print x, testVec

    for x in ["dogo", "meño"]:
        neiVec = skmodel.get_nearest_neighbors(x)
        print x, neiVec

    # skmodel.get_analogies("berlin", "germany", "france")
    with open(sammpleFile, 'r') as f1:
        for line in f1:
            tks = line.strip().rstrip().split(" ")
            for item in tks:
                if wordMap.has_key(item):
                    wordMap[item] = wordMap[item] + 1
                else :
                    wordMap[item] = 1

        print "word num:%d" %(len(wordMap))

    with open(vecFile, 'w') as f2:
        for k,y in wordMap.items():
            printLine = ",".join([k, ",".join(list(map(lambda x: str(x),  skmodel.get_word_vector(x) )))])
            f2.write(printLine + "\n")

    skmodel.save_model(modelFile)

def main(argv):
    # reload(sys)
    # sys.setdefaultencoding("utf-8")

    if len(argv) != 3 :
        print("arg err: len = %d" %(len(argv)))
        print("arg like: python sample.txt model.bin vector.txt")
    else:
        print("go main : %d" %(len(argv)))

    fasttext_embedding(argv[1], argv[2], argv[3])

if __name__=="__main__":
    main(sys.argv)
