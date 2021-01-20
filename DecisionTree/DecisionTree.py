import pandas as pd
import numpy as np


class Node:
    def __init__(self,key):
        self.left = None
        self.right = None
        self.val = key
        self.atr1 = None #classification 1 in the attribute root
        self.atr2 = None #classification 2 in the attribute root
        self.atr1_label1 = None #the reuslt label that with the majority vote if root goes down in atr1
        self.atr1_label2 = None
        self.atr2_label1 = None #the reuslt label that with the majority vote if root goes down in atr2
        self.atr2_label2 = None
        self.num_atr1_label1 = 0 #the number of labels that with the majorty vote if root goes down in atr1
        self.num_atr1_label2 = 0
        self.num_atr2_label1 = 0 #the number of labels that with the majorty vote if root goes down in atr2
        self.num_atr2_label2 = 0




#read data from destinated folder by argument infile(str)
def read_train_data(infile: str):
    with open( infile,'r' ) as f:
        contents = pd.read_csv( f,delimiter='\t' )
    return contents

#helper funciton to get gini impurity for a leaf node
def get_gini(data1: pd.DataFrame,data2: pd.DataFrame):
    num_atr1 = data1.shape[0]
    num_atr2 = data2.shape[0]
    total_num = num_atr1 + num_atr2
    if (total_num) ==0:
        return 0
    prob_atr1 = float( num_atr1 / total_num )
    prob_atr2 = float( num_atr2 / total_num )
    gini = 1 - prob_atr1 ** 2 - prob_atr2 ** 2
    return gini

#split dataset into two by its specified index(int) (containing binary classification)
def split_data_by_label(data: pd.DataFrame,index:int):
    if data.size == 0:
        return pd.DataFrame(),pd.DataFrame()
    atr1 = data.iloc[:,index].unique()[0]
    try:
        atr2 = data.iloc[:,index].unique()[1]
        return data[data.iloc[:,index] == atr1],data[data.iloc[:,index] == atr2]
    except:
        return data[data.iloc[:,index] == atr1],pd.DataFrame()

#select decision stump by choosing the attribute with the lowest gini impurity
def creat_stump(data: pd.DataFrame,left:int):
    #stop the recursion when any of the situaiton comes
    # 1) it is the leaf node -> exhaust all attributes
    # 2)read the maximum depth
    if data.size ==0 or left ==0:
        return None
    #calculate gini impurity for all allowed attributes and store them in a master list: dic(list)
    dic = []
    for i in range( len( data.columns ) - 1 ):
        data1,data2 = split_data_by_label( data,i )
        data3,data4 = split_data_by_label( data1,-1 )
        gini1 = get_gini( data3,data4 )
        data5,data6 = split_data_by_label( data2,-1 )
        gini2 = get_gini( data5,data6 )
        weighted_gini = data1.shape[0] / data.shape[0] * gini1 + data2.shape[0] / data.shape[0] * gini2
        dic.append(weighted_gini)
    #pick the attribute with the lowest gini impurity and make it the next node
    #calculate related information about the new node
    #1) result labels this node: self.atr1, self.atr2(optional)
    #2) within the subset splited by the chosen node's attribute label, target's labels with the majority votes: self.atr1_label1, self.atr2.label1
    root = Node(data.columns[dic.index(min(dic))])
    data5,data6 = split_data_by_label(data,dic.index(min(dic)))
    label,count = np.unique(data5.iloc[:,-1],return_counts = True)
    root.atr1 = data[root.val].unique()[0]
    root.num_atr1 = data[data[root.val] == root.atr1].shape[0]
    root.atr1_label1 =label[np.where(count==np.max(count))[0]][0]
    root.num_atr1_label1 = count[np.where(count==np.max(count))[0]][0]
    if label.size > 1: #check whether goes down into atr1 perfectly classify the result
                        # / if there is other result label if we goes down into atr1
        root.atr1_label2 = label[np.where( count == np.min( count ) )[0]][0]
        root.num_atr1_label2 = count[np.where( count == np.min( count ) )[0]][0]
    if data6.shape[0] > 0: #check whether root node perfectly classify the result/ if there is other result label in root node
        root.atr2 = data[root.val].unique()[1]
        root.num_atr2 = data[data[root.val] == root.atr2].shape[0]
        label,count = np.unique( data6.iloc[:,-1],return_counts=True )
        root.atr2_label1 = label[np.where( count == np.max( count ) )[0]][0]
        root.num_atr2_label1 = count[np.where( count == np.max( count ) )[0]][0]
        if label.size > 1:#check whether goes down into atr2 perfectly classify the result
                        # / if there is other result label, go down into atr2
            root.atr2_label2 = label[np.where( count == np.min( count ) )[0]][0]
            root.num_atr2_label2 = count[np.where( count == np.min( count ) )[0]][0]
    else: #do nothing if root node has already perfectly classified the result -> becomes a leaf node
        return
    data1,data2 = split_data_by_label(data,dic.index(min(dic))) #split data by the attribute root.val
    try:
        #drop the root.val attribute and go on to create another stump
        #store the returning root from the following steps as child node (left or right)
        root.left = creat_stump(data1.drop([root.val],axis = 1),left-1)
        root.right = creat_stump(data2.drop([root.val],axis = 1),left-1)
    except:
        pass
    return root

def write_label(data: pd.DataFrame, root:Node):
    # helps to make prediction by adding the prediction column into the original dataframe
    # we can make final prediction only when we go down to leaf node
    # 1) iterate the tree to the leaf node; 2) label the data based on the tree 3) write the prediction as a new column in dataframe
    if root ==None: #check whether the input is a valid decision tree
        return None
    data1 = data[data[root.val] == root.atr1]
    data3 = write_label( data1,root.left ) #goes left until it reaches to leaf node
    data2 = pd.DataFrame(columns = data1.columns)
    data4 = pd.DataFrame(columns = data1.columns)
    if root.atr2 != None: #check whether root has right node (more than one classification of attribute root.val)
        data2 = data[data[root.val] == root.atr2]
        data4 = write_label( data2,root.right) #goes right if there exist another classification of attribute root.val
    if root.left == None and root.right == None: #check whether this is a leaf node
        pd.options.mode.chained_assignment = None
        data1['Prediction'] = [root.atr1_label1 for _ in range( data1.shape[0] )] #append prediction for data falls in classification 1 of attribute root.val
        data2['Prediction'] = [root.atr2_label1 for _ in range( data2.shape[0] )]  #append prediction for data falls in classification 2 of attribute root.val
        return pd.concat([data1,data2]).reset_index(drop=True) #return the combined dataframe (the whole dataframe that goes into root node)
    return pd.concat([data3,data4]).reset_index(drop=True) #pass along the dataframe return from leaf node up to the root of the tree

def report_error(data:pd.DataFrame):
    data['error'] = (data.iloc[:,-1] != data.iloc[:,-2])
    error_rate = data['error'].sum() / data.shape[0]
    return error_rate

def report_tree(root:Node,count:int):
    if root ==None:
        return
    print('{}{} = {} [{} {} /{} {}]'.format('|'*count,root.val,root.atr1,root.atr1_label1,root.num_atr1_label1,root.atr1_label2,root.num_atr1_label2))
    report_tree(root.left,count+1)
    print('{}{} = {} [{} {} /{} {}]'.format( '|' * (count),root.val,root.atr2,root.atr2_label1,root.num_atr2_label1,root.atr2_label2,root.num_atr2_label2 ) )
    report_tree( root.right,count+1)


if __name__ == "__main__":
    raw = read_train_data('train.tsv' )
    max = min( 3,raw.shape[1] - 1 ) #contrain the maximum depth of the tree
    label,count = np.unique( raw.iloc[:,-1],return_counts=True )
    print( '[{} {}/{} {}]'.format(label[0],count[0],label[1],count[1])) #report the overall structure of the input data based on result label
    root = creat_stump(raw,max) #train tree
    newData = write_label(raw,root) #make prediciton based on the tree
    report_tree(root,1)
    print( report_error( newData ) )


