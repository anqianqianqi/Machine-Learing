import sys
import pandas as pd

#read training dataset from command line parameters
#return a dataframe with training dataset
def read_train_date(infile:str):
    with open(infile,'r') as f:
        contents = pd.read_csv(f,delimiter='\t')
    return contents

#split dataset based on selected attribute
def split_data(contents:pd.DataFrame,index:int):
    left = contents[(contents.iloc[:,index] == 'y')|(contents.iloc[:,index] == 'A')]
    right = contents[(contents.iloc[:,index] == 'n')|(contents.iloc[:,index] == 'notA')]
    return left,right

#create decision stump by majority vote rule
def create_stump(data,atr1,atr2):
    vote1 = data[data.iloc[:,-1]==atr1].count()[0]
    vote2 = data[data.iloc[:,-1]==atr2].count()[0]
    if vote1 > vote2:
        return atr1
    else:
        return atr2

#write output file containing error rate for the training model
def write_outfile(index:int,test_data_path:str,outfile:str,infile:str):
    with open(test_data_path,'r') as f:
        test_data = pd.read_csv(f,delimiter='\t')
    contents = read_train_date(infile)
    atr1 = contents.iloc[:, -1].unique()[0]
    atr2 = contents.iloc[:, -1].unique()[1]
    left, right = split_data(contents, index)
    left_vote = create_stump(left,atr1,atr2)
    right_vote = create_stump(right,atr1,atr2)
    output = [left_vote if line[index]=='y' else right_vote for line in test_data.values]
    with open(outfile, 'w') as f:
        for line in output:
            f.write(str(line+'\n'))
    test_data['prediction'] = output
    test_data['error'] = (test_data.iloc[:,-1]!=test_data.iloc[:,-2])
    error_rate = test_data['error'].sum()/test_data.shape[0]
    return error_rate




if __name__ == '__main__':
    infile = sys.argv[1]
    #test with test data
    print("error rate at testing set is {}".format(write_outfile(int(sys.argv[3]), sys.argv[4], sys.argv[2],sys.argv[1])))
    #test with train data
    print("error rate at training set is {}".format(write_outfile(int(sys.argv[3]), sys.argv[1], sys.argv[5],sys.argv[1])))
