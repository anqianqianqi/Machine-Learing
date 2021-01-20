import csv
import re

def get_dic():
    infile = 'dict.txt'
    with open( infile,'r' ) as f:
        lines = csv.reader( f,delimiter='\n' )
        master_dict = {line[0].split(' ')[0]:line[0].split(' ')[1] for line in [item for item in lines]}
    return master_dict

def get_features(infile:str, outfile:str, master_dict: dict,flag:int):
    output = open( outfile,'w' )
    with open( infile,'r' ) as f:
        raw = csv.reader( f,delimiter='\n' )
        for longline in raw:
            item = longline[0].split('\t')
            line = str(item[0])
            occured_words = {}
            for word in re.split(r'[^A-Za-z]',item[1]):
                if word.strip() in master_dict.keys():
                    occured_words[master_dict[word.strip()]] = occured_words.get(master_dict[word.strip()],0) + 1
            if flag == 1:
                for wordindex,value in occured_words.items():
                    line += '\t' + str(wordindex) + ":" + str(1)
                line+='\n'
                output.write(line)
            elif flag == 2:
                for wordindex,value in occured_words.items():
                    if value < 4:
                        line += '\t' + str(wordindex) + ":" + str(1)
                        line+='\n'
                        output.write(line)
                        
get_features('train_data.tsv','result.csv',get_dic(),1)