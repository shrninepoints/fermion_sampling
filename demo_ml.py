import fermion_sampling as fs
import numpy as np
import os
import gensim

VEC_PATH = "/Users/macbookpro/projects/FermionSampling/train_vec"
L_PATH = "/Users/macbookpro/projects/FermionSampling/train_l"
TEST_PATH = "/Users/macbookpro/projects/FermionSampling/test_data"
model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=20) # Doc2Vec model
d = 10 # dimension of feature vector
theta = np.array([0.00030181 ,-0.02927969  ,0.02378645 ,-0.01102074 ,-0.00431801 ,-0.00295598,0.00050187  ,0.00323944 , 0.00947083  ,0.01119595])

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
    if method == 'dpp':    
        result = fs.dpp(v,number,size)
    elif method == 'sampling':
        result = fs.sampling(v,number,size)
    result = m2x(result)
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
    file_paths = os.listdir(folder_path)[0:num]
    ts_list = [readData(folder_path + "/" + path) for path in file_paths]
    return ts_list

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
    step = 1/1000000
    for i in range(1000):
        grad = gradLiklihood(oracleSummary(ts_list),theta)
        theta = theta + step * grad
        print("theta = ",theta)
    return theta

def selectionMBR(sample_num,text,summary_sentence_num):
    summary_list = []
    L = generateL(text,theta)
    e,v = np.linalg.eig(L)
    e = e
    for i in range(sample_num):
        result = sampleL(e,v,"sampling")
        print(result)
        summary = [text[i] for i in result]
        summary = ''.join(summary)
        summary_list.append(summary)
    print(summary_list)
    summary_list_vec = [vector(summary) for summary in summary_list]
    similarity_list = []
    for i in range(sample_num):
        similarity = np.sum([cosine(summary_list_vec[i],summary_list_vec[j]) for j in range(sample_num)])
        similarity_list.append(similarity)
    selection = np.argmin(similarity_list)
    return summary_list[selection]

if __name__ == "__main__":
    vecTrain()
    ts_list = readMultipleData(VEC_PATH,5)
    text,_ = ts_list[0]
    print(selectionMBR(5,text,3))
    '''
    theta_init = theta
    print(thetaTrain(theta))
    print(theta_init)
    
    L = generateL(text,theta)
    e, v = np.linalg.eig(L)
    print(e)
    print(sampleL(e,v,"sampling"))
    #print(vector("Some scoff at the notion that movies do anything more than entertain . They are wrong . Sure , it 's unlikely that one movie alone will change your views on issues of magnitude . But a movie -LRB- or TV show -RRB- can begin your `` education '' or `` miseducation '' on a topic . And for those already agreeing with the film 's thesis , it can further entrench your views ."))
    #print(readData("/Users/macbookpro/projects/FermionSampling/cnn_stories_tokenized/0a0a4c90d59df9e36ffec4ba306b4f20f3ba4acb.story"))
    '''