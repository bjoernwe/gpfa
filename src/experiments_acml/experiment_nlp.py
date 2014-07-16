import numpy as np
import scipy.sparse

from matplotlib import pyplot

#import nltk
from nltk.corpus import brown

if __name__ == '__main__':
    
    # download corpus
    #nltk.download('brown')
    
    # read corpus
    excluded_tags = set([u'.', u',', u':', u'``', u"''", u'(', u')', u"'"])
    words = [word[0].lower() for word in brown.tagged_words()[:100000] if word[1][0] not in excluded_tags]
    words_set = list(set(words))
    N = len(words_set)
    print N, 'unique words in Brown corpus.'
    
    # construct transition matrix
    words_indices = [words_set.index(word) for word in words]
    print 1
    word_indices_row = words_indices[:-1]# + words_indices[1:]
    print 2
    word_indices_col = words_indices[1:]# + words_indices[:-1]
    print 3
    T = scipy.sparse.coo_matrix((np.ones(len(word_indices_row)), (word_indices_row, word_indices_col)), shape=(N,N), dtype=int)
    #T = T.todok()
    print 4
    T = T.tolil()
    print 5
    
    for i in range(5):
        print words_set[i]
        for j in T[i,:].nonzero()[1]:
            print '->', words_set[j], T[i,j]
    
    non_zero_indices_row, non_zero_indices_col = T.nonzero()
    
    
    #W = W.todense()
    #print W
    #max_transition = np.unravel_index(np.argsort(W.flat)[-11], W.shape)
    #print words_set[max_transition[0]], words_set[max_transition[1]]
    
    #pyplot.imshow(W)
    #pyplot.show()
    