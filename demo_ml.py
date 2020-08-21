import fermion_sampling as fs
import numpy as np
import os
import gensim
import time

VEC_PATH = "/Users/macbookpro/projects/FermionSampling/train_vec"
L_PATH = "/Users/macbookpro/projects/FermionSampling/train_l"
TEST_PATH = "/Users/macbookpro/projects/FermionSampling/test_data"
model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=20) # Doc2Vec model
d = 10 # dimension of feature vector
method = "dpp"
article_length = 60
sample_num = 100
theta = np.array([0.00096383 ,-0.01216535 , 0.01574886 ,-0.00411098 ,-0.00392975, -0.00080255,0.01070928 , 0.00381164 , 0.01000864, -0.00190676])

def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args[0:1], kw, te-ts))
        return result
    return timed

def sampleL(e,v,method): # sample with input L
    def m2x(result):
        x = []
        for i in range(len(result)):
            if result[i] == 1:
                x.append(i)
        return x
    size = len(e)
    removal = []
    for i in range(size):
        if np.random.rand() > (e[i]/(e[i]+1)):
            removal.append(i)
    v = np.delete(v,removal,1)
    number = v.shape[1]
    try:
        if method == 'dpp':    
            result = fs.dpp(np.dot(v,v.T.conj()),number,size)
        elif method == 'sampling':
            result = fs.sampling(v,number,size)
        result = m2x(result)
    except ValueError:
        result = None
    return result # result gives a list of the index of sentence selected

def cosine(a,b): # cosine similarity between two vectors
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def readMultipleData(folder_path,num):
    def readData(path): # return list of strings in form of text-summary pair
        with open(path) as t:
            lines = t.readlines()
            lines = [i for i in lines if not i == '\n']
            sep = lines.index('@highlight\n')
            text = lines[0:sep]
            summary = [lines[sep+2*i+1] for i in range((len(lines) - sep) // 2)]
        return text,summary       
    file_paths = os.listdir(folder_path)
    #file_paths.sort()
    file_paths = file_paths[0:num]
    ts_list = [readData(folder_path + "/" + path) for path in file_paths]
    return ts_list

def readRandomData(path):
    with open(path) as t:
        lines = t.readlines()
        text = lines[0].split(". ")
        text = [sentence + "\n" for sentence in text]
        text = text[0:article_length]
        summary = []
    return text,summary
    
def vecTrain():
    ts_list = readMultipleData(VEC_PATH,100)
    train_list = [t for (t,s) in ts_list]
    train_corpus = []
    for i in range(len(train_list)):
        doc = ''.join(train_list[i])
        tokens = gensim.utils.simple_preprocess(doc)
        train_corpus.append(gensim.models.doc2vec.TaggedDocument(tokens, [i]))
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

def vector(sentence): # with a pre-trained model, vectorize a sentence
    return model.infer_vector(gensim.utils.simple_preprocess(sentence))

def matrixS(text_vec): # return S matrix for a text TODO: add hyperparameter to increase replusion
    l = len(text_vec)
    S = np.zeros((l,l))
    for i in range(l):
        for j in range(l):
            S[i,j] = cosine(text_vec[i],text_vec[j])
        S[i] = S[i] / np.linalg.norm(S[i])
    return S

def featureFunction(text): # return a d-dim vector for each sentence, in size*d matrix
    # TODO: add more features
    length = len(text)
    const = np.array([1 for i in range(length)])# propornational to sample size
    len_list = np.array([len(sentence) for sentence in text]) # longer sentence has higher weight
    position_mark = np.array([abs(length/2-i) // 5 for i in range(length)]) # sentence at front/end has higher weight
    pronouns_num = np.array([i.count("I ") + i.count("he ") + i.count("she ") + i.count("He ") + i.count("She ") + i.count("You ") + i.count("you ") for i in text])
    feature = np.array([const,len_list,len_list ** 2 / 100,len_list ** 3 / 10000,position_mark,position_mark ** 2,position_mark ** 3,pronouns_num,pronouns_num ** 2,pronouns_num ** 3]).T
    return feature

def qualityVector(feature,theta): # return quality vector in 1*size
    q = [np.dot(feature[i],theta) for i in range(len(feature))]
    q = np.clip(q,-10,10)
    q = np.exp(q)
    return q

def generateL(text,theta):
    text_vec = [vector(line) for line in text]
    S = matrixS(text_vec)
    q = qualityVector(featureFunction(text),theta)
    Q = np.diag(q)
    L = Q.dot(S).dot(Q)
    return L

def oracleSummary(ts_list): # return the index of sentence in oracle summary (not summary itself)
    o_list = []
    for t,s in ts_list:
        o = []
        t_vec_list = [vector(sentence) for sentence in t]
        s_vec_list = [vector(sentence) for sentence in s]
        for s_vec in s_vec_list:
            similarity = [cosine(t_vec,s_vec) for t_vec in t_vec_list]
            o.append(np.argmin(similarity)) # TODO: remove repeat selected sentence
        o_list.append(o)
    t_list,_ = list(zip(*ts_list))
    to_list = list(zip(t_list,o_list))
    return to_list # list of (text, summary index) pairs

def thetaTrain(theta):
    def gradLiklihood(to_list,theta):
        sum_grad = np.zeros(d)
        for t,o in to_list:
            L = generateL(t,theta)
            e,v = np.linalg.eig(L)
            Kii = np.zeros(len(e))
            for i in range(len(e)):
                temp = np.array([np.square(v[i,n]) * e[n] / (e[n]+1) for n in range(len(e))])
                Kii[i] = np.sum(temp) 
            feature = featureFunction(t)
            grad = np.sum(feature[o],0) - Kii.dot(feature)
            sum_grad = sum_grad + grad
        return sum_grad
    step = 1/10000
    to_list = oracleSummary(ts_list)
    for i in range(1000):
        grad = gradLiklihood(to_list,theta)
        theta = theta + step * grad
        print("theta = ",theta)
    return theta

@timeit
def selectionMBR(sample_num,text,min_length,max_length):
    summary_list = []
    L = generateL(text,theta)
    e,v = np.linalg.eig(L)
    print("article length = ", len(e))
    print(method)
    samples = 0
    output_length = 0
    while samples < sample_num:
        result = sampleL(e,v,method) # result could be none or a result
        if result and len(result) > min_length and len(result) < max_length: # remove results that are None and too long/short
            output_length = output_length + len(result)
            result.sort()
            #print(result)
            summary = [text[i] for i in result]
            summary = ''.join(summary)
            summary_list.append(summary)
            samples = samples + 1
    output_length = output_length / sample_num
    summary_list_vec = [vector(summary) for summary in summary_list]
    similarity_list = []
    for i in range(len(summary_list)):
        similarity = np.sum([cosine(summary_list_vec[i],summary_list_vec[j]) for j in range(len(summary_list))])
        similarity_list.append(similarity)
    selection = np.argmin(similarity_list)
    print("output length = ",output_length)
    return summary_list[selection]

if __name__ == "__main__":
    vecTrain()
    #ts_list = readMultipleData(VEC_PATH,1)
    text,_ = readRandomData("/Users/macbookpro/projects/FermionSampling/test_data/00000test.story")
    print(selectionMBR(sample_num,text,0,200))
    '''
    theta_init = theta
    print("new theta = ",thetaTrain(theta))
    print("previous_theta =",theta_init)
    
    L = generateL(text,theta)
    e, v = np.linalg.eig(L)
    print(e)
    print(sampleL(e,v,"sampling"))
    '''