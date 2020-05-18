from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk
import networkx as nx
import tensorflow as tf
import numpy as np
import random
import gensim 
from gensim.models import Word2Vec 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
class team2box:
    
    def __init__(self,dataset):
        self.N=0 # number of nodes in the CQA network graph N=|Qestions|+|Answers|+|Experts|
        self.G={}
        self.qnum=0
        self.anum=0
        self.enum=0
        self.dataset=dataset
        self.load_graph()
        #self.save_qraph()
        
        #self.walks()
        
        
      
            
    def save_qraph(self):
        qfile=open(self.dataset+"/krnmdata1/CQAG.txt","w")
        qfile.write("N="+str(self.N)+" Questions= "+str(self.qnum)+" index=0.."+str(self.qnum-1)
                    +"; Answers= "+str(self.anum)+" index="+str(self.qnum)+".."+str(self.qnum+self.anum-1)
                    +"; Experts= "+str(self.enum)+" index="+str(self.qnum+self.anum)+".."+str(self.qnum+self.anum+self.enum-1)+"\n")
        
        for node in self.G:
            for i in range(len(self.G[node]['n'])):
                if node< self.G[node]['n'][i]:
                    qfile.write(str(node)+" "+str(self.G[node]['n'][i])+" "+str(self.G[node]['w'][i])+"\n")
        qfile.close()
    
    def load_graph(self):
        qpfile=open(self.dataset+"/krnmdata1/questionposts.txt")
        qpfile.readline()
        line=qpfile.readline().strip()
        qids=[]
        aids=[]
        eids=[]
        while line:
            qp=line.split(" ")
            qid=int(qp[0].strip())            
            if qid not in qids:
                qids.append(qid)
            caids=qp[1::2] 
            for aid in caids:
                if int(aid) not in aids:
                    aids.append(int(aid))
            line=qpfile.readline().strip()    
        qpfile.close()  
        print(len(qids))
        print(len(aids))
        pufile=open(self.dataset+"/krnmdata1/postusers.txt")
        pufile.readline()
        line=pufile.readline().strip()
        while line:
            ids=line.split(" ")
            eid=int(ids[1].strip())            
            if eid not in eids:
                eids.append(eid)
            line=pufile.readline().strip() 
        pufile.close()
        print(len(eids))
        
        self.qnum, self.anum, self.enum=len(qids), len(aids), len(eids)
        self.N=len(qids)+len(aids)+len(eids)
        
        #create CQA network graph
        qpfile=open(self.dataset+"/krnmdata1/questionposts.txt")
        qpfile.readline()
        line=qpfile.readline().strip()        
        while line:
            qp=line.split(" ")
            qid=qids.index(int(qp[0].strip()))           
            if qid not in self.G:
                self.G[qid]={'n':[],'w':[]}
                
            caids=qp[1::2] 
            #print(caids)
            caidsscore=qp[2::2] 
            #print(caidsscore)
            for ind in range(len(caids)):
                aid=aids.index(int(caids[ind]))+len(qids)
                if aid not in self.G:
                    self.G[aid]={'n':[qid],'w':[int(caidsscore[ind])]}
                self.G[qid]['n'].append(aid)
                self.G[qid]['w'].append(int(caidsscore[ind]))
            line=qpfile.readline().strip()    
        qpfile.close() 
        pufile=open(self.dataset+"/krnmdata1/postusers.txt")
        pufile.readline()
        line=pufile.readline().strip()
        while line:
            ids=line.split(" ")
            aid=aids.index(int(ids[0].strip()))+len(qids)
            eid=eids.index(int(ids[1].strip()))+len(qids)+len(aids)           
                      
            if eid not in self.G:
                self.G[eid]={'n':[aid],'w':[self.G[aid]['w'][0]]}
                
            else:
                self.G[eid]['n'].append(aid)
                self.G[eid]['w'].append(self.G[aid]['w'][0])
            self.G[aid]['n'].append(eid)
            self.G[aid]['w'].append(self.G[aid]['w'][0])    
            line=pufile.readline().strip() 
        pufile.close()
        
        #print(self.G)
        
    def walker(self,start, walklen):
        walk=""
        ii=0 
        s1=start
        #start=random.randint(self.qnum+self.anum,self.N) # start from expert
        prev=start
        while ii<walklen: 
            #print("st="+ str(start)+" pre="+str(prev))            
            ind=0
            if len(self.G[start]['n'])==1:
                neib=self.G[start]['n']
                #print(neib)
                ind=0  
            else:
                weights=self.G[start]['w'].copy()  
                neib=self.G[start]['n'].copy()
                #print(neib)
                #print(weights)
                if prev in neib:
                    indpre=neib.index(prev)                
                    del weights[indpre:indpre+1]
                    del neib[indpre:indpre+1]
                    #print(neib)
                    #print(weights)
                if len(neib)==1:
                    ind=0
                else:    
                    sumw=sum(weights)                
                    ranw=random.randint(1,sumw)
                    #print("sumw="+str(sumw)+" ranw="+str(ranw))                
                    for i in range(len(neib)):
                        if ranw<=sum(weights[0:i+1]):
                            ind=i
                            break
                        
            walk+= " "+str(start)   
           # if len(self.G[start]['n'])==1:
           #     break
            prev=start
            start=neib[ind]
            
                
            #if start>self.qnum+self.anum:
            ii+=1
        #if s1==0:
        #    print(walk)
        return walk.strip()    
    
    def walks(self,walklen):
        G=nx.Graph();
        G=nx.read_weighted_edgelist(self.dataset+"/krnmdata1/CQAG1.txt")
        rw = BiasedRandomWalk(StellarGraph(G))

        weighted_walks = rw.run(
        nodes=G.nodes(), # root nodes
        length=walklen,    # maximum length of a random walk
        n=10,          # number of random walks per root node 
        p=0.5,         # Defines (unormalised) probability, 1/p, of returning to source node
        q=2.0,         # Defines (unormalised) probability, 1/q, for moving away from source node
        weighted=True, #for weighted random walks
        seed=42        # random seed fixed for reproducibility
        )
        print("Number of random walks: {}".format(len(weighted_walks)))
        print(weighted_walks[0:10])
               
        return weighted_walks       

    def getembedding(self,walklen):
        walks=self.walks(walklen)
        random.shuffle(walks)
        print (walks[0:20])
        model = gensim.models.Word2Vec(walks, min_count = 1, size = 32, window =9,sg=1) 
        model.save(self.dataset+"/krnmdata1/"+"team2box.model")
        print("done!")
        #print("Cosine similarity between '0' " +  "and '863'  : ", model.similarity('0', '863'))
        #print("Cosine similarity between '0' " +  "and '1863'  : ", model.similarity('0', '1863'))
    
    def analyze(self,zz):
        model = gensim.models.Word2Vec.load(self.dataset+"/krnmdata1/"+"team2box2.model")
        print(self.G[zz]['n'])
        print(self.G[zz]['w'])
        minnib=1
        list1=list(range(0,self.qnum,1))
        list1.extend(list(range(self.qnum+self.anum,self.N,1)))
        nib=[]
        for i in self.G[zz]['n']:
            nib.extend(self.G[i]['n'])
        print(nib)
        for i in nib:
            if i in list1 and i!=zz:
                print("Cosine similarity between "+ str(zz) +  " and "+str(i)+"  : ", model.similarity(str(zz), str(i)))
                if model.similarity(str(zz), str(i))<minnib:
                    minnib=model.similarity(str(zz), str(i))
        maxsim=0
        
        
        for i in list1:
            if i!=zz:
                sim=model.similarity(str(zz), str(i))
                if sim>maxsim:
                    maxsim=sim
                if sim>=minnib:
                    print(sim," i=",i)
                    print(self.G[i]['n'])
                    print(self.G[i]['w'])
        print(maxsim)        

    def getsim(self,i,data):
        model = gensim.models.Word2Vec.load(self.dataset+"/krnmdata1/"+"team2box.model")
        for x in data:
            print("Cosine similarity between "+ str(i) +  " and "+str(x)+"  : ", model.similarity(str(i), str(x)))
    
    def getembeddingvector(self):
        embedding=[]
        model = gensim.models.Word2Vec.load(self.dataset+"/krnmdata1/"+"team2box.model")
        for i in range(self.qnum):
            embedding.append(model.wv[str(i)])
        for i in range(self.anum):
            embedding.append(model.wv[str(i)])    
        for i in range(self.enum):
            embedding.append(model.wv[str(self.qnum+self.anum+i)])    
        return np.array(embedding)
    
    def saveembeddingvector(dataset):
        pfile=open(dataset+"/krnmdata1/properties.txt")
        pfile.readline()
        properties=pfile.readline().strip().split(" ")
        pfile.close()
        N=int(properties[0]) # number of nodes in the CQA network graph N=|Qestions|+|Answers|+|Experts|                
        qnum=int(properties[1])
        anum=int(properties[2])
        enum=int(properties[3])
        embedding=[]
        model = gensim.models.Word2Vec.load(dataset+"/krnmdata1/"+"team2box.model")
        for i in range(qnum):
            embedding.append(model.wv[str(i)])
        for i in range(enum):
            embedding.append(model.wv[str(qnum+anum+i)])    
        np.savetxt(dataset+"/krnmdata1/n2v_expert_question_w1_embedding3.txt",np.array(embedding), fmt='%f')
        
    def visualize1(self,list1):
        y=np.loadtxt(self.dataset+"/krnmdata1/"+"embeding_tsne.txt")
        plt.figure(figsize=(15,15))
        plt.plot(y[list1,0],y[list1,1],'r+');        
        for i in list1:
            plt.text(y[i,0],y[i,1], i, fontsize=8)
        
    def visualize(self):
        self.embedding=self.getembeddingvector()
        #qeembed=np.delete(self.embedding, [range(self.qnum, self.qnum+self.anum, 1)], 0)
        qeembed=self.embedding
        model = TSNE(n_components=2, random_state=0)
        y=model.fit_transform(qeembed) 
        np.savetxt(self.dataset+"/krnmdata1/"+"embeding_tsne.txt", y)
        plt.figure(figsize=(15,15))
        plt.plot(y[0:self.qnum,0],y[0:self.qnum,1],'r+');
        
        for i in range(self.qnum):
            plt.text(y[i,0],y[i,1], i, fontsize=4)
        
        plt.plot(y[self.qnum+self.anum:self.N,0],y[self.qnum+self.anum:self.N,1],'b*');
        
        for i in range(self.enum):
            plt.text(y[self.qnum+self.anum+i,0],y[self.qnum+self.anum+i,1], self.qnum+self.anum+i, fontsize=4)    
        plt.show();
        
dataset=["android.stackexchange.com","history/history.stackexchange.com","dba/dba.stackexchange.com/","physics","example_cqa","example2_cqa"]
ob=team2box(dataset[1])
ob.getembedding(20)
team2box.saveembeddingvector(dataset[1])
#ob.analyze()
#ob.visualize()
print("done!")