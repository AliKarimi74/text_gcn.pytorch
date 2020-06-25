import os
import random
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
import sys
from utils import print_log, loadWord2Vec, clean_str

# check arguments
if len(sys.argv) != 2:
    sys.exit("Use: python build_graph.py <dataset>")

datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']
# build corpus
dataset = sys.argv[1]

if dataset not in datasets:
    sys.exit("wrong dataset name")

# Read Word Vectors
# word_vector_file = 'data/glove.6B/glove.6B.300d.txt'
# word_vector_file = 'data/corpus/' + dataset + '_word_vectors.txt'
# _, embd, word_vector_map = loadWord2Vec(word_vector_file)
# word_embeddings_dim = len(embd[0])

word_embeddings_dim = 300
word_vector_map = {}

# create and shuffle train and test set
doc_name_list = []
doc_train_list = []
doc_test_list = []

print_log("reading dataset and cleaning ...")
# read index file
with open('data/' + dataset + '.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        doc_name_list.append(line.strip())
        temp = line.split("\t")
        if temp[1].find('test') != -1:
            doc_test_list.append(line.strip())
        elif temp[1].find('train') != -1:
            doc_train_list.append(line.strip())
# print(doc_train_list)
# print(doc_test_list)

doc_content_list = []
with open('data/corpus/' + dataset + '.clean.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        doc_content_list.append(line.strip())
# print(doc_content_list)


def document_ids(doc_lst):
    id_list = []
    for name in doc_lst:
        id_list.append(doc_name_list.index(name))
    random.shuffle(id_list)
    return id_list


print_log("shuffling and partitioning dataset ...")
train_ids = document_ids(doc_train_list)
# print(train_ids)
# partial labeled data
# train_ids = train_ids[:int(0.2 * len(train_ids))]
train_ids_str = '\n'.join(str(index) for index in train_ids)
with open('data/' + dataset + '.train.index', 'w') as f:
    f.write(train_ids_str)

test_ids = document_ids(doc_test_list)
# print(test_ids)

test_ids_str = '\n'.join(str(index) for index in test_ids)
with open('data/' + dataset + '.test.index', 'w') as f:
    f.write(test_ids_str)

ids = train_ids + test_ids
# print(ids)
# print(len(ids))

# write shuffle data into the files
shuffle_doc_name_list = []
shuffle_doc_words_list = []
for id in ids:
    shuffle_doc_name_list.append(doc_name_list[int(id)])
    shuffle_doc_words_list.append(doc_content_list[int(id)])

shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list)
shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)
with open('data/' + dataset + '_shuffle.txt', 'w') as f:
    f.write(shuffle_doc_name_str)
with open('data/corpus/' + dataset + '_shuffle.txt', 'w') as f:
    f.write(shuffle_doc_words_str)

print_log("building vocab and word document list ...")
# build vocab
word_freq = {}
word_set = set()
for doc_words in shuffle_doc_words_list:
    words = doc_words.split()
    for word in words:
        word_set.add(word)
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

vocab = list(word_set)
vocab_size = len(vocab)

# find word --- documents edges in the graph
word_doc_list = {}
for i in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    appeared = set()
    for word in words:
        if word in appeared:
            continue
        if word in word_doc_list:
            doc_list = word_doc_list[word]
            doc_list.append(i)
            word_doc_list[word] = doc_list
        else:
            word_doc_list[word] = [i]
        appeared.add(word)

word_doc_freq = {}
for word, doc_list in word_doc_list.items():
    word_doc_freq[word] = len(doc_list)

word_id_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i

# save vocab file
vocab_str = '\n'.join(vocab)
with open('data/corpus/' + dataset + '_vocab.txt', 'w') as f:
    f.write(vocab_str)

'''
Word definitions begin
'''
'''
definitions = []

for word in vocab:
    word = word.strip()
    synsets = wn.synsets(clean_str(word))
    word_defs = []
    for synset in synsets:
        syn_def = synset.definition()
        word_defs.append(syn_def)
    word_des = ' '.join(word_defs)
    if word_des == '':
        word_des = '<PAD>'
    definitions.append(word_des)

string = '\n'.join(definitions)


f = open('data/corpus/' + dataset + '_vocab_def.txt', 'w')
f.write(string)
f.close()

tfidf_vec = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf_vec.fit_transform(definitions)
tfidf_matrix_array = tfidf_matrix.toarray()
print(tfidf_matrix_array[0], len(tfidf_matrix_array[0]))

word_vectors = []

for i in range(len(vocab)):
    word = vocab[i]
    vector = tfidf_matrix_array[i]
    str_vector = []
    for j in range(len(vector)):
        str_vector.append(str(vector[j]))
    temp = ' '.join(str_vector)
    word_vector = word + ' ' + temp
    word_vectors.append(word_vector)

string = '\n'.join(word_vectors)

f = open('data/corpus/' + dataset + '_word_vectors.txt', 'w')
f.write(string)
f.close()

word_vector_file = 'data/corpus/' + dataset + '_word_vectors.txt'
_, embd, word_vector_map = loadWord2Vec(word_vector_file)
word_embeddings_dim = len(embd[0])
'''
'''
Word definitions end
'''

# label list
label_set = set()
for doc_meta in shuffle_doc_name_list:
    temp = doc_meta.split('\t')
    label_set.add(temp[2])
label_list = list(label_set)

label_list_str = '\n'.join(label_list)
with open('data/corpus/' + dataset + '_labels.txt', 'w') as f:
    f.write(label_list_str)

# x: feature vectors of training docs, no initial features
# select 90% training set
train_size = len(train_ids)
val_size = int(0.1 * train_size)
real_train_size = train_size - val_size
# different training rates

real_train_doc_names = shuffle_doc_name_list[:real_train_size]
real_train_doc_names_str = '\n'.join(real_train_doc_names)
with open('data/' + dataset + '.real_train.name', 'w') as f:
    f.write(real_train_doc_names_str)


def create_data_arrays(size, start_idx=0, only_return_list_data=False):
    row_x = []
    col_x = []
    data_x = []
    for i in range(size):
        doc_vec = np.zeros(word_embeddings_dim)
        doc_words = shuffle_doc_words_list[i + start_idx]
        words = doc_words.split()
        doc_len = len(words)
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                doc_vec = doc_vec + np.array(word_vector)

        for j in range(word_embeddings_dim):
            row_x.append(i)
            col_x.append(j)
            # np.random.uniform(-0.25, 0.25)
            data_x.append(doc_vec[j] / doc_len)  # doc_vec[j] / doc_len

    y = []
    for i in range(size):
        doc_meta = shuffle_doc_name_list[i + start_idx]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        y.append(one_hot)

    if only_return_list_data:
        return row_x, col_x, data_x, y

    # x = sp.csr_matrix((test_size, word_embeddings_dim), dtype=np.float32)
    x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(size, word_embeddings_dim))
    y = np.array(y)

    return x, y


# train input and target
print_log("creating train data x, y ...")
x, y = create_data_arrays(real_train_size)

# test input and target
# tx: feature vectors of test docs, no initial features
print_log("creating test data x, y ...")
test_size = len(test_ids)
tx, ty = create_data_arrays(test_size, train_size)

# allx: the the feature vectors of both labeled and unlabeled training instances
# (a superset of x)
# unlabeled training instances -> words

# init word vectors
word_vectors = np.random.uniform(-0.01, 0.01,
                                 (vocab_size, word_embeddings_dim))

for i in range(len(vocab)):
    word = vocab[i]
    if word in word_vector_map:
        vector = word_vector_map[word]
        word_vectors[i] = vector

# merge all document and word vectors into one 2d array
row_allx, col_allx, data_allx, ally = create_data_arrays(train_size, 0, True)

for i in range(vocab_size):
    for j in range(word_embeddings_dim):
        row_allx.append(int(i + train_size))
        col_allx.append(j)
        data_allx.append(word_vectors.item((i, j)))

for i in range(vocab_size):
    one_hot = [0 for l in range(len(label_list))]
    ally.append(one_hot)

row_allx = np.array(row_allx)
col_allx = np.array(col_allx)
data_allx = np.array(data_allx)

allx = sp.csr_matrix(
    (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))

ally = np.array(ally)

print('x: ', x.shape, 'y:', y.shape,
      '\ntx: ', tx.shape, 'ty:', ty.shape,
      '\nvocab_size: ', vocab_size,
      '\nall_x: ', allx.shape, 'all_y:', ally.shape)

'''
Doc word heterogeneous graph
'''

print_log("creating document word heterogeneous graph ...")
# calculate word co-occurrence with context windows
window_size = 20
windows = []

# extract all windows
for doc_words in shuffle_doc_words_list:
    words = doc_words.split()
    length = len(words)
    if length <= window_size:
        windows.append(words)
    else:
        # print(length, length - window_size + 1)
        for j in range(length - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)
            # print(window)

word_window_freq = {}
for window in windows:
    appeared = set()
    for i in range(len(window)):
        if window[i] in appeared:
            continue
        if window[i] in word_window_freq:
            word_window_freq[window[i]] += 1
        else:
            word_window_freq[window[i]] = 1
        appeared.add(window[i])

word_pair_count = {}
for window in windows:
    for i in range(1, len(window)):
        for j in range(0, i):
            word_i = window[i]
            word_i_id = word_id_map[word_i]
            word_j = window[j]
            word_j_id = word_id_map[word_j]
            if word_i_id == word_j_id:
                continue
            word_pair_str = str(word_i_id) + ',' + str(word_j_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            # two orders
            word_pair_str = str(word_j_id) + ',' + str(word_i_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1

# graph matrix data holders
# weight matrix segmentation details:
# x and y segments: [ .. train_size .. ] => train data
#                   [ .. vocab_size .. ] => words data
#                   [ .. test_size ... ] test data
row = []
col = []
weight = []

# pmi as weights
# fill weight [train_size : train_size + vocab_size][train_size : train_size + vocab_size] block
num_window = len(windows)
for key in word_pair_count:
    temp = key.split(',')
    i = int(temp[0])
    j = int(temp[1])
    count = word_pair_count[key]
    word_freq_i = word_window_freq[vocab[i]]
    word_freq_j = word_window_freq[vocab[j]]
    pmi = log((1.0 * count / num_window) /
              (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
    if pmi <= 0:
        continue
    row.append(train_size + i)
    col.append(train_size + j)
    weight.append(pmi)

# word vector cosine similarity as weights
'''
for i in range(vocab_size):
    for j in range(vocab_size):
        if vocab[i] in word_vector_map and vocab[j] in word_vector_map:
            vector_i = np.array(word_vector_map[vocab[i]])
            vector_j = np.array(word_vector_map[vocab[j]])
            similarity = 1.0 - cosine(vector_i, vector_j)
            if similarity > 0.9:
                print(vocab[i], vocab[j], similarity)
                row.append(train_size + i)
                col.append(train_size + j)
                weight.append(similarity)
'''

# doc word frequency
# fill weight [0:train_size][train_size : train_size + vocab_size] block for train set
# and  weight [train_size + vocab_size :][train_size : train_size + vocab_size] block for test set
doc_word_freq = {}

for doc_id in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[doc_id]
    words = doc_words.split()
    for word in words:
        word_id = word_id_map[word]
        doc_word_str = str(doc_id) + ',' + str(word_id)
        if doc_word_str in doc_word_freq:
            doc_word_freq[doc_word_str] += 1
        else:
            doc_word_freq[doc_word_str] = 1

# calculate tf-idf score for doc-word edges
for i in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_word_set = set()
    for word in words:
        if word in doc_word_set:
            continue
        j = word_id_map[word]
        key = str(i) + ',' + str(j)
        freq = doc_word_freq[key]
        if i < train_size:
            row.append(i)
        else:
            row.append(i + vocab_size)
        col.append(train_size + j)
        idf = log(1.0 * len(shuffle_doc_words_list) /
                  word_doc_freq[vocab[j]])
        weight.append(freq * idf)
        doc_word_set.add(word)

# create adjacency matrix
node_size = train_size + vocab_size + test_size
adj = sp.csr_matrix(
    (weight, (row, col)), shape=(node_size, node_size))

print_log('saving all data ...')
# dump objects
with open("data/ind.{}.x".format(dataset), 'wb') as f:
    pkl.dump(x, f)

with open("data/ind.{}.y".format(dataset), 'wb') as f:
    pkl.dump(y, f)

with open("data/ind.{}.tx".format(dataset), 'wb') as f:
    pkl.dump(tx, f)

with open("data/ind.{}.ty".format(dataset), 'wb') as f:
    pkl.dump(ty, f)

with open("data/ind.{}.allx".format(dataset), 'wb') as f:
    pkl.dump(allx, f)

with open("data/ind.{}.ally".format(dataset), 'wb') as f:
    pkl.dump(ally, f)

with open("data/ind.{}.adj".format(dataset), 'wb') as f:
    pkl.dump(adj, f)
