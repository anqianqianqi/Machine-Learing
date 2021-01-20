import learnhmm
import numpy as np
import time

def predict(words,tags_index,words_index,initial_words,A,B):
    predict_tags = ""
    errors = 0
    total_words = 0
    log_likelihood = 0.0 # use log likelihood to prevent underflow
    for line in words:
        line = line.split(sep=" ")
        T = len(line)
        tags_size = tags_index.size
        alpha = np.zeros((tags_size,T))
        #initialize alpha0
        word,tag = line[0].split( sep="_" )
        Xt = np.where( words_index == word )[0]
        alpha[:,0] =  initial_words*B[Xt]
        #loop all timestamps in forward order
        for t in range(1,T):
            word,tag = line[t].split( sep="_" )
            Xt = np.where( words_index == word )[0]
            #loop all possible tag k in this timestamp t > 1
            for k in range(tags_size):
                Bjk = B[Xt].T[k]
                #add all possible previous tag j given current tag k * alpha(t-1) in the previous timestamp t-1
                sum = np.sum(alpha[:,t-1] * A[:,k])
                alpha[k].T[t] = Bjk*sum
        beta = np.zeros((tags_size,T))
        beta[:,-1] = 1
        #loop all timestamps in backward order
        for t in range(T-2,-1,-1):
            word,tag = line[t+1].split( sep="_" )
            Xt = np.where( words_index == word )[0]
            #loop all possible tag k in this timestamp t
            for k in range(tags_size):
                #add all possible following tag j given current tag k * beta(t+1) in the following timestamp t+1
                beta[k].T[t] = np.sum(beta[:,t+1] * A[k]*B[Xt])
        # Predict tags for this sentence
        for t in range(T):
            word,tag = line[t].split( sep="_" )
            multiple = alpha[:,t] * beta[:,t]
            # np.argmax deals with tie by taking the one of first appearance
            tag_max = tags_index[np.argmax( multiple )]
            #tag_max = tags_index[np.where(multiple == np.max(multiple))[0][0]]
            predict_tags += word+ "_"+ tag_max + " "
            total_words += 1
            if (tag!=tag_max): errors+=1
        predict_tags += "\n"
        # compute P(X) by adding alpha at end point: all_paths_weights
        all_paths_weights = np.sum( alpha[:,-1] )
        # Compute log likelihood of P(X)
        log_likelihood += np.log(all_paths_weights)
    return predict_tags,float(errors/total_words), float(log_likelihood/words.size)

if __name__ == '__main__':
    tags_index,words_index = learnhmm.read_index()
    words = learnhmm.read_data( "trainwords.txt" )
    #A: (tagj,tagk) tag j is followed by tag k: tag j -> tag k
    #B: (Xt,Yt)
    #initial_words: (prb of tag j as a start of the sentence, )
    initial_words,A,B = learnhmm.count(words,words_index,tags_index)
    test_words = learnhmm.read_data("testwords.txt")
    start_time = time.time()
    result = predict(test_words,tags_index,words_index,initial_words,A,B)
    print(result[0])
    print( "Error rate at test is {}".format(result[1] ))
    print( "Log likelihood of P(X) is {}".format(result[2] ))
    print("---{}minutes---".format((time.time()-start_time)/60.0))
