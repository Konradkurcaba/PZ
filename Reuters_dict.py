from string import digits

import scipy
import re
from nltk.corpus import reuters
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

def target_matrix(docs_id):           #create target sparse matrix from given docs_id
    iteration_counter = 0
    categories = reuters.categories() # contains all categories
    target_array = np.empty(shape=[0, 90])
    for i in docs_id:
        categories_list = reuters.categories(i) #return categories for i (doc_id)
        arr = np.zeros((90,), dtype=np.int) #create array with 90 zeros
        for j in categories_list:   # j - one of doc-i category
            counter = 0
            for c in categories:
                if j == c:  # if j category == category in categories_list
                    arr[counter] = 1 #put 1 in array
                counter = counter + 1
            if iteration_counter == 0:
                train_target_array = arr
                iteration_counter = +1

        target_array = np.vstack((target_array, arr))


    new_array = scipy.sparse.csr_matrix(target_array)  # create sparse array from dense array
    return new_array

micro_avg_denominators = []
micro_avg_numerators = []

def recall_precision(TP,FP,FN,index):

    precision = TP[index] / (TP[index] + FP[index])
    p = precision
    recall = TP[index] / (TP[index] + FN[index])
    r = recall
    if (precision > recall):
        p_denominator = (TP[index] + FP[index]) *2  #save precision denominator
        r_denominator = (TP[index] + FN[index] ) *2 #save recall denominator
        while p != r:
            p_denominator +=1
            r_denominator -=1
            p = (TP[index] *2 )/ p_denominator
            r = (TP[index] *2 ) / r_denominator

    micro_avg_denominators.append(p_denominator)
    micro_avg_numerators.append(TP[index] * 2)
    return precision,recall,p


documents = reuters.fileids()


categories = reuters.categories();

train_docs = [] #docs array
train_docs_ids = [] #docs_id_array
test_docs = []
test_docs_ids = []


for doc_id in reuters.fileids():
    if doc_id.startswith("train"):
        string = reuters.raw(doc_id)
        remove_digits = str.maketrans('', '', digits)
        res = string.translate(remove_digits) # remove numbers
        res = res.replace(",","") #remove dots
        res = res.replace(".","") #remove commas
        train_docs.append(res) #get train docs
        train_docs_ids.append(doc_id)   #get_id

    else:
        string = reuters.raw(doc_id)
        remove_digits = str.maketrans('', '', digits)
        res = string.translate(remove_digits)  # remove numbers
        res = res.replace(",", "")  # remove dots
        res = res.replace(".", "")  # remove commas
        test_docs.append(res)  # get train docs
        test_docs_ids.append(doc_id)  # get_id

count_vect = CountVectorizer(stop_words="english") #use english stopwords
dictionary = count_vect.fit_transform(train_docs) #create dictionary for train docs - sparse matrix
test_dictionary = count_vect.transform(test_docs) #create dictionary for train docs - sparse matrix

tfidf_transformer = TfidfTransformer()
transformed_dict = tfidf_transformer.fit_transform(dictionary) #create transformed dictionary using tf_idf
test_transformed_dict = tfidf_transformer.fit_transform(test_dictionary)

#TF_IDF of term is
#tf-idf(d, t) = tf(t) * idf(d, t)
#tf(t) = term_frequency/all_terms in given document
#idf(d, t) = log [ (1 + n) / (1 + df(d, t)) ] + 1      // df(d,t) number of documents contain term t


train_target=target_matrix(train_docs_ids)
test_target=target_matrix(test_docs_ids)


clf = OneVsRestClassifier(LinearSVC(random_state=0))
clf.fit(transformed_dict, train_target)



#clf = OneVsRestClassifier(MultinomialNB()).fit(transformed_dict, train_target)

#list = clf.estimators_
#est0 = list[0]
#est0.class_log_pior = np.array([1,0])


results = clf.predict(test_transformed_dict) #predict results - sparse matrix






TP = np.zeros(90,dtype=np.int) #array with True positive values for each class
FP = np.zeros(90,dtype=np.int) #array with False positive values for each class
FN = np.zeros(90,dtype=np.int) #array with False negative values for each class
TN = np.zeros(90,dtype=np.int) #array with True negative values for each class


for i in range(0,3019):
    result_row = results.getrow(i).toarray()  #one row from result matrix
    test_target_row = test_target.getrow(i).toarray() #one row from target matrix
    for j in range(0,90):
        result = np.take(result_row,j)  #value with index j
        target = np.take(test_target_row,j) #value with index j

        if  result == 1  and target == 1:
            TP[j] +=1
        if result == 0 and target == 1:
            FN[j] +=1
        if result == 1 and target == 0:
            FP[j] +=1
        if result == 0 and target == 0:
            TN[j] +=1

print("earn: ")
precision,recall,p = recall_precision(TP,FP,FN,21)
print ("precision: ")
print (precision)
print ("recall: ")
print (recall)
print ("Precision Recall Breakeven point: " )
print (p)

print("acq: ")
precision,recall,p = recall_precision(TP,FP,FN,0)
print ("precision: ")
print (precision)
print ("recall: ")
print (recall)
print ("Precision Recall Breakeven point: " )
print (p)

print("money-fx: ")
precision,recall,p = recall_precision(TP,FP,FN,46)
print ("precision: ")
print (precision)
print ("recall: ")
print (recall)
print ("Precision Recall Breakeven point: " )
print (p)

print("grain: ")
precision,recall,p = recall_precision(TP,FP,FN,26)
print ("precision: ")
print (precision)
print ("recall: ")
print (recall)
print ("Precision Recall Breakeven point: " )
print (p)

print("crude: ")
precision,recall,p = recall_precision(TP,FP,FN,17)
print ("precision: ")
print (precision)
print ("recall: ")
print (recall)
print ("Precision Recall Breakeven point: " )
print (p)

print("trade: ")
precision,recall,p = recall_precision(TP,FP,FN,84)
print ("precision: ")
print (precision)
print ("recall: ")
print (recall)
print ("Precision Recall Breakeven point: " )
print (p)

print("interest: ")
precision,recall,p = recall_precision(TP,FP,FN,34)
print ("precision: ")
print (precision)
print ("recall: ")
print (recall)
print ("Precision Recall Breakeven point: " )
print (p)

print("ship: ")
precision,recall,p = recall_precision(TP,FP,FN,71)
print ("precision: ")
print (precision)
print ("recall: ")
print (recall)
print ("Precision Recall Breakeven point: " )
print (p)

print("wheat: ")
precision,recall,p = recall_precision(TP,FP,FN,86)
print ("precision: ")
print (precision)
print ("recall: ")
print (recall)
print ("Precision Recall Breakeven point: " )
print (p)


print("corn: ")
precision,recall,p = recall_precision(TP,FP,FN,12)
print ("precision: ")
print (precision)
print ("recall: ")
print (recall)
print ("Precision Recall Breakeven point: " )
print (p)

n_count = 0
d_count = 0
for i in range (0,10):
    n_count += micro_avg_numerators[i]
    d_count += micro_avg_denominators[i]

print("MicroAVG: ")
print(n_count/d_count)




#print("train end")
#print("score:")
#print(clf.score(test_transformed_dict, test_target))

