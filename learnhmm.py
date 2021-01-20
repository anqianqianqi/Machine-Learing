import numpy as np

def read_data(infile:str):
    with open(infile,'r') as f:
        words = np.array(f.read().split(sep="\n"))
    return words

def read_index():
    path_tags = "index_to_tag.txt"
    path_words = "index_to_word.txt"
    with open(path_tags,'r') as f:
        tags_index = np.array(f.read().split(sep="\n"))
    with open(path_words,'r') as f:
         words_index = np.array(f.read().split(sep="\n"))
    return tags_index,words_index

def count(words,words_index,tags_index):
    tags_size = tags_index.size
    words_size = words_index.size
    # add pseudocount 1 to every item
    initial_words = np.ones( tags_size )
    A = np.ones( tags_size * tags_size )
    B = np.ones( tags_size * words_size )
    for line in words:
        line = line.split( " " )
        start = line[0]
        start_tag = start.split( sep="_" )[1]
        # for each word, count the number of time it appears as an initial word in a sentence
        initial_words[np.where( tags_index == start_tag )] += 1
        line_tag = np.array( [] )
        for word in line:
            wordextract,tag = word.split( sep="_" )
            Bj = np.where( tags_index == tag )[0]
            Bk = np.where( words_index == wordextract )[0]
            # count the number of times tag j (Sj) is associated with word k
            B[Bk * tags_size + Bj] += 1
            line_tag = np.append( line_tag,[tag] )
        for i in range( line_tag.size - 1 ):
            Aj = np.where( tags_index == line_tag[i] )[0]
            Ak = np.where( tags_index == line_tag[i + 1] )[0]
            # count the number of times tag j (Sj) is followed by tag k (Sk): tag j -> tag k
            A[Aj * tags_size + Ak] += 1
    A = A.reshape( tags_size,tags_size )
    B = B.reshape(words_size,tags_size)
    #transform counts to probabilities
    Asumlarge = np.array( [np.sum( A,axis=1 )] * tags_size ).T
    Bsumlarge = np.array( [np.sum(B,axis = 0)] * words_size )
    initial_words = initial_words/np.sum(initial_words)
    A = A / Asumlarge
    B = B / Bsumlarge
    return initial_words,A,B

