import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import collections
from collections import Counter 

nltk.download('wordnet')

def extract_data():
    # Loading data
    dataset = pd.read_csv('emails.csv')
    print(dataset .columns) #Index(['text', 'spam'], dtype='object')
    print(dataset.shape)  #(5728, 2)

    #Pre-processing

    #Checking for duplicates and removing them
    dataset.drop_duplicates(inplace = True)
    print(dataset.shape)  #(5695, 2)

    #Checking for any null entries in the dataset
    print (pd.DataFrame(dataset.isnull().sum()))

    #Removing subject:
    dataset['text']=dataset['text'].map(lambda text: text[9:])
    #Removing numerical values
    dataset['text'] = dataset['text'].apply(lambda x: re.sub(r'\d+(\.\d+)?', '',x))
    #Removing punctuator
    dataset['text'] = dataset['text'].apply(lambda x: re.sub(r'[^\w\d\s]', ' ',x))
    #Removing whitespace
    dataset['text'] = dataset['text'].apply(lambda x: re.sub(r'\s+', ' ',x))
    #Removing leading and trailing whitespace
    dataset['text'] = dataset['text'].apply(lambda x: re.sub(r'^\s+|\s+?$', '',x))
    #Removing stopwords
    stop_words = ['_','i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    dataset['text'] = dataset['text'].apply(lambda x: ' '.join(
        term for term in x.split() if term not in set(stop_words))
    )
    #Removing one and two letter words 
    dataset['text'] = dataset['text'].apply(lambda x: re.sub(r'\b\w{1,2}\b','',x))
    new_list=list(dataset['text'])
    label=list(dataset['spam'])
    return new_list,label
    
new_list,label = extract_data()
count_spam=len([i for i in label if i==1])
count_ham=len([i for i in label if i==0])
print("number of spamand ham resp.",count_spam,count_ham)


# Making dictionary of importent words
#def Make_dict(new_list):
split_list=[]
for i in range(len(new_list)):
    split_list.append(new_list[i].split())

str2 = [] 
spam_diction={}
ham_diction={}
diction={}
lemmatizer = WordNetLemmatizer()
# loop till string values present in list str 
for j in range(len(new_list)):              
    for i in range(len(split_list[j])):
        # lemmatize data
        split_list[j][i]=lemmatizer.lemmatize(split_list[j][i])
        if split_list[j][i] not in str2: 
            # insert value in str2 
            str2.append(split_list[j][i])
            spam_diction[split_list[j][i]]=0
            ham_diction[split_list[j][i]]=0
            diction[split_list[j][i]]=1
            if(label[j]==1):
                spam_diction[split_list[j][i]]=1
            else:
                ham_diction[split_list[j][i]]=1
        else:
            diction[split_list[j][i]]+=1
            if(label[j]==1):
                spam_diction[split_list[j][i]]+=1
            else:
                ham_diction[split_list[j][i]]+=1


spam_diction_c=spam_diction.copy()
ham_diction_c=ham_diction.copy()

for i in spam_diction_c.copy():
    if spam_diction_c[i] < 25: 
        spam_diction_c.pop(i)
for i in ham_diction_c.copy():
    if ham_diction_c[i] < 50: 
        ham_diction_c.pop(i)
diction_c=Counter(spam_diction_c)+Counter(ham_diction_c)

print("length of spam", len(spam_diction_c))
print("length of ham", len(ham_diction_c))
print("total number of used words ",len(diction_c))
t_s=sum(spam_diction_c.values())
t_h=sum(ham_diction_c.values())
t_total=t_s+t_h


# funtion to preprocess data
def pre_process_data(dataset):
    dataset =re.sub(r'\d+(\.\d+)?', '',dataset)
    dataset = re.sub(r'[^\w\d\s]', ' ',dataset)
    dataset= re.sub(r'\s+', ' ',dataset)
    dataset = re.sub(r'^\s+|\s+?$', '',dataset)
    dataset = re.sub(r'\b\w{1,2}\b','',dataset)
    dataset = dataset.lower()
    data_list=list(dataset)
    lemmatizer = WordNetLemmatizer()
    split_list=[]
    split_list=dataset.split()
    for i in range(len(split_list)):
        split_list[i]=lemmatizer.lemmatize(split_list[i])
    
    return split_list
	

#fetching data from first test email
file1 = open("./test/email1.txt","r")
data= file1.read()
test1=pre_process_data(data)
prior_spam=0.5
prior_ham=0.5
posterior_spam=prior_spam
posterior_ham=prior_ham
for i in test1:  
    #print(i)
    Clist=Counter(spam_diction_c)+ Counter(ham_diction_c)
    if i in Clist.copy():
        if i in spam_diction_c.copy():
            #print('#',np.log((spam_diction_c[i]+1)/(count_spam+2)))
            posterior_spam+=np.log((spam_diction_c[i]+1)/(count_spam+2))
        else: 
            #print('##',np.log(1/(count_spam+2)))
            posterior_spam+=np.log(1/(count_spam+2))

        if i in ham_diction_c.copy():
            #print('*',np.log((ham_diction_c[i]+1)/(count_ham+2)))
            posterior_ham+=np.log((ham_diction_c[i]+1)/(count_ham+2))
        else: 
            #print('**',np.log(1/(count_ham+2)))
            posterior_ham+=np.log(1/(count_ham+2))

        #word_prob[i]=(diction[i]+t_total)/(t_total*(1+len(diction)))
posterior_spam=posterior_spam*prior_spam
posterior_ham=posterior_ham*prior_ham
print(posterior_spam,posterior_ham)
if(posterior_spam>posterior_ham):
    print('spam')
else:
    print('ham')

	
#fetching data from second test email
file2 = open("./test/email2.txt","r")
data= file2.read()
test2=pre_process_data(data)
prior_spam=0.5
prior_ham=0.5
posterior_spam=0
posterior_ham=0
for i in test2:   
    #print(i)
    Clist=Counter(spam_diction_c)+ Counter(ham_diction_c)
    if i in (Clist):
        if i in spam_diction_c.copy():
            #print('#',np.log((spam_diction_c[i]+1)/(count_spam+2)))
            posterior_spam+=np.log((spam_diction_c[i]+1)/(count_spam+2))
        else: 
            #print('##',np.log(1/(count_spam+2)))
            posterior_spam+=np.log(1/(count_spam+2))

        if i in ham_diction_c.copy():
            #print('*',np.log((ham_diction_c[i]+1)/(count_ham+2)))
            posterior_ham+=np.log((ham_diction_c[i]+1)/(count_ham+2))
        else: 
            #print('**',np.log(1/(count_ham+2)))
            posterior_ham+=np.log(1/(count_ham+2))

posterior_spam=posterior_spam*prior_spam
posterior_ham=posterior_ham*prior_ham
print(posterior_spam,posterior_ham)
if(posterior_spam>posterior_ham):
    print('spam')
else:
    print('ham')
	

def check_spam(data):
    test=pre_process_data(data)
    prior_spam=0.5
    prior_ham=0.5
    posterior_spam=0
    posterior_ham=0
    for i in test:   
        #print(i)
        Clist=Counter(spam_diction_c)+ Counter(ham_diction_c)
        if i in (Clist):
            if i in spam_diction_c.copy():
                #print('#',np.log((spam_diction_c[i]+1)/(count_spam+2)))
                posterior_spam+=np.log((spam_diction_c[i]+1)/(count_spam+2))
            else: 
                #print('##',np.log(1/(count_spam+2)))
                posterior_spam+=np.log(1/(count_spam+2))

            if i in ham_diction_c.copy():
                #print('*',np.log((ham_diction_c[i]+1)/(count_ham+2)))
                posterior_ham+=np.log((ham_diction_c[i]+1)/(count_ham+2))
            else: 
                #print('**',np.log(1/(count_ham+2)))
                posterior_ham+=np.log(1/(count_ham+2))

    posterior_spam=posterior_spam*prior_spam
    posterior_ham=posterior_ham*prior_ham
    #print(posterior_spam,posterior_ham)
    if(posterior_spam>posterior_ham):
        return 1
    else:
        return 0

# accuracy test
dataset = pd.read_csv('spam_ham.csv',encoding= 'latin-1')
print(dataset .columns) #Index(['text', 'spam'], dtype='object')
print(dataset.shape)  #(5728, 2)
l=list(dataset['v1'])
for i in range(len(dataset)):
    if l[i]=='spam':
        label[i]=1
    else:
        label[i]=0
d=list(dataset['v2'])
print(d[0])

count=0
new_lable=[]
for i in range(len(dataset)):
    gg=check_spam(d[i])
    if(label[i]==gg):
        count+=1

print('correct classified mails',count)      
