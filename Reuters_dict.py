import scipy
import nltk
from sklearn.metrics import precision_score, recall_score

nltk.download('reuters')
from string import digits
from nltk.corpus import reuters
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from scipy import stats
import matplotlib.pyplot as plt


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
    return new_array, target_array

precision_avg = []
recall_avg = []
def recall_precision(TP,FP,FN,index):

    precision = TP[index] / (TP[index] + FP[index])
    recall = TP[index] / (TP[index] + FN[index])
    precision_avg.append(precision)
    recall_avg.append(recall)

    return precision,recall




def breakEvenPoint(classNumber, all_probability_matrix, dense_target_matrix):
    resultForOneCategory = []
    targetResultForOneCategory = []
    for i in range(0, 3019):
        resultForOneCategory.append(all_probability_matrix[i, classNumber])
        targetResultForOneCategory.append(dense_target_matrix[i,classNumber])
    precision_list = [] #for threshold from 0.0  to 1.0
    recall_list = []    #for threshold from 0.0  to 1.0
    result_list = []    #for threshold from 0.0  to 1.0
    p = 0.0

    for i in range(0, 100):

        for j in range(0, 3019):

            if resultForOneCategory[j] > p: #if probability is bigger than threshold
                result_list.append(1)
            else:
                result_list.append(0)       #if isn't
        precision_list.append(precision_score(result_list, targetResultForOneCategory))
        recall_list.append(recall_score(result_list, targetResultForOneCategory))
        result_list.clear()
        p = p + 0.01

    for i in range(0, 100):
        prec = round(precision_list[i],3)
        if round(precision_list[i],3) == round(recall_list[i],3):
            return (round(precision_list[i],3))






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

train_target, xd = target_matrix(train_docs_ids)
test_target, dense_test_target = target_matrix(test_docs_ids)

#clf = OneVsRestClassifier(LinearSVC(random_state=0)) #Linear Support Vector Classification.
#clf = OneVsRestClassifier(AdaBoostClassifier(MultinomialNB(alpha=0.05),n_estimators=100)) #Naive Bayes with AdaBoost
clf = OneVsRestClassifier(MultinomialNB(alpha=0.05)) #Naive Bayes
#clf = OneVsRestClassifier(SVC(probability=True, kernel='linear')) # C-Support Vector Classification.
#clf = OneVsRestClassifier(AdaBoostClassifier(SVC(probability=True, kernel='linear'),n_estimators=25 ))

clf.fit(transformed_dict, train_target)

results = clf.predict(test_transformed_dict) #predict results - sparse matrix


all_propability_matrix = clf.predict_proba(test_transformed_dict);




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
precision,recall = recall_precision(TP,FP,FN,21)
print ("precision: ")
print (precision)
print ("recall: ")
print (recall)
print ("break even point:")
print (breakEvenPoint(21,all_propability_matrix,dense_test_target))

print("acq: ")
precision,recall = recall_precision(TP,FP,FN,0)
print ("precision: ")
print (precision)
print ("recall: ")
print (recall)


print("money-fx: ")
precision,recall = recall_precision(TP,FP,FN,46)
print ("precision: ")
print (precision)
print ("recall: ")
print (recall)


print("grain: ")
precision,recall = recall_precision(TP,FP,FN,26)
print ("precision: ")
print (precision)
print ("recall: ")
print (recall)


print("crude: ")
precision,recall = recall_precision(TP,FP,FN,17)
print ("precision: ")
print (precision)
print ("recall: ")
print (recall)


print("trade: ")
precision,recall = recall_precision(TP,FP,FN,84)
print ("precision: ")
print (precision)
print ("recall: ")
print (recall)


print("interest: ")
precision,recall = recall_precision(TP,FP,FN,34)
print ("precision: ")
print (precision)
print ("recall: ")
print (recall)


print("ship: ")
precision,recall = recall_precision(TP,FP,FN,71)
print ("precision: ")
print (precision)
print ("recall: ")
print (recall)


print("wheat: ")
precision,recall = recall_precision(TP,FP,FN,86)
print ("precision: ")
print (precision)
print ("recall: ")
print (recall)



print("corn: ")
precision,recall = recall_precision(TP,FP,FN,12)
print ("precision: ")
print (precision)
print ("recall: ")
print (recall)


print("Precision AVG:")
print(sum(precision_avg) / len(precision_avg ))
print("Recall AVG:")
print(sum(recall_avg) / len(recall_avg))