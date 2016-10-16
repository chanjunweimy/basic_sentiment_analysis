import csv 
from os import listdir 
from os.path import isfile, join

mypath = ".";
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for file in files:    
    
    if file.endswith(".csv") == False:
        continue;
    
    sentences = []        
    with open(file, 'rb') as csvfile:      
        spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')         
        for row in spamreader:                        
            isChosen = True                        
            for column in row:                                
                if column == "start_time":                                        
                    isChosen = False                                        
                    break                               
                elif column == "Ellie":  
                    isChosen = False                                        
                    break                                                
            if not row:                                
                continue                       
            elif isChosen:                                
                sentences.append(row[-1])                    
            
    extractedFileName = file.replace(".csv", ".txt")        
    extractedFile = open(extractedFileName, "w")        
    for sentence in sentences:                
        extractedFile.write(sentence + "\n")                
    extractedFile.close()                