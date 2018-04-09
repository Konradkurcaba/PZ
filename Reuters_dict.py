import scipy
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




documents = reuters.fileids()


categories = reuters.categories();

train_docs = [] #docs array
train_docs_ids = [] #docs_id_array
test_docs = []
test_docs_ids = []

for doc_id in reuters.fileids():
    if doc_id.startswith("train"):
        train_docs.append(reuters.raw(doc_id)) #get train docs
        train_docs_ids.append(doc_id)   #get_id
    else:
        test_docs.append(reuters.raw(doc_id)) #get test docs
        test_docs_ids.append(doc_id) #get id

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

clf = OneVsRestClassifier(LinearSVC(random_state=0)).fit(transformed_dict, train_target)
#clf = OneVsRestClassifier(MultinomialNB()).fit(transformed_dict, train_target)

results = clf.predict(test_transformed_dict) #predict results - sparse matrix





TP = np.zeros(90,dtype=np.int)
FP = np.zeros(90,dtype=np.int)

FN = np.zeros(90,dtype=np.int)



for i in range(0,3019):
    result_row = results.getrow(i).toarray()
    test_target_row = test_target.getrow(i).toarray()
    for j in range(0,90):
        result = np.take(result_row,j)
        target = np.take(test_target_row,j)

        if  result == 1  and target == 1:
            TP[j] +=1
        if result == 0 and target == 1:
            FN[j] +=1
        if result == 1 and target == 0:
            FP[j] +=1

print("earn: ")
precision = TP[21] /( TP[21] + FP[21])
recall = TP[21] / (TP[21] + FN[21])
print (precision)
print (recall)

print("corn: ")
precision = TP[12] /( TP[12] + FP[12])
recall = TP[12] / (TP[12] + FN[12])
print (precision)
print (recall)





print("dalej")


#print("train end")
#print("score:")
#print(clf.score(test_transformed_dict, test_target))

