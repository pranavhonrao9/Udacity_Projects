
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
plotly.offline.init_notebook_mode(connected=True)


# In[2]:


training_df = pd.read_csv('~/train.csv', error_bad_lines=False)


# In[3]:


training_df.shape


# In[4]:


test_df = pd.read_csv('~/test.csv')


# In[3]:


test_df.shape


# In[5]:


training_df.info()


# In[6]:


feature_df =training_df['is_duplicate']


# In[7]:


training_df =training_df.drop(columns=['is_duplicate'])


# In[8]:


training_error1 =training_df['question1']


# In[9]:


training_error2 =training_df['question2']


# In[11]:


#training_error1.head()


# In[12]:


#training_error2.head()


# In[10]:


questions_training_df = training_df[['question1','question2']]


# In[11]:


questions_test_df = test_df[['question1','question2']]


# In[12]:


temp_check = pd.concat([questions_training_df, questions_test_df])


# In[16]:


from nltk.corpus import stopwords
from nltk.stem.porter import *
import nltk
nltk.download("stopwords")
nltk.download('wordnet')


# In[17]:


sw_dict={}
    
for i in stopwords.words('english'):
    sw_dict[i]=1
    


# In[1]:


import time


# In[19]:


# In[20]:


import re
from nltk.stem import WordNetLemmatizer
#from nltk.stem import PorterStemmer
from stemming.porter2 import stem

lemi_words=[]
stem_words=[]
lmtzr = nltk.WordNetLemmatizer().lemmatize
ps = PorterStemmer()



def do_lemitise(word):
    old_word=word
    word = lmtzr(word)
    if word != old_word:
        lemi_words.append((old_word, word))
        
    return word
    
def do_stem(word):
    old_word=word
    word = stem(word)
    if word != old_word:
        stem_words.append((old_word, word))
    return word
        

def do_process_word(word):
    word = do_stem(word)
    word = do_lemitise(word)
    return word
    
    
    

def corpus_filteration(dfList):
    start=time.time()
    
    actual_list =[]
    

    
    item_num=0

    for k, v in dfList.iteritems():
        questions_corpus = re.sub('[^a-zA-Z]',' ', str(v)).lower().split()
        '''
        for word in questions_corpus:
            if word not in sw_dict.keys():
                #old_word=word
                word = lmtzr(word)
                #if word != old_word:
                    #lemi_words.append((old_word, word))
                questions_corpus.append(word)
        '''
                
                
                
            
        questions_corpus = [do_process_word(word) for word in questions_corpus if word not in sw_dict.keys()]
        
        actual_list.append(questions_corpus)
        item_num = item_num + 1
        if (item_num%1000000 == 0):
            print ("Done so far {} ".format(item_num) )
    elapsed =time.time() - start
    return actual_list      
    
    


# In[19]:


training_error1.head()


# In[21]:


training_error1_cf = corpus_filteration(training_error1)


# In[22]:


training_error2_cf = corpus_filteration(training_error2)


# In[23]:


temp_check_test_question1 = temp_check['question1']


# In[24]:


test_question1_list1 = corpus_filteration(temp_check_test_question1)


# In[24]:


temp_check_test_question2 = temp_check['question2']


# In[25]:


test_question2_list1 = corpus_filteration(temp_check_test_question2)


# In[26]:


test_question_list= test_question1_list1 + test_question2_list1


# In[27]:


from gensim.models import Word2Vec
model = Word2Vec(test_question_list, min_count=1,size=200)
# summarize the loaded model
print(model)
# summarize vocabulary
words = list(model.wv.vocab)
#print(words)
# access vector for one word
#print(model['hack'])
# save model
model.save('model.bin')
# load model
#new_model = Word2Vec.load('model.bin')
#print(new_model)


# In[28]:


#print(model['step'])


# In[29]:



from gensim.models import Word2Vec
from gensim.models import KeyedVectors
#filename = 'GoogleNews-vectors-negative300.bin'
#import pyemd
#word2vec_model = Word2Vec.load(filename)
word2vec_model = KeyedVectors.load('model.bin')
word2vec_model.init_sims(replace=True)


# In[ ]:





# In[ ]:





# In[30]:


from pyemd import emd
def distance_calculation(d1,d2):
        i=j=0
        distance_list=[]
        len_train_01 =len(d1)
        len_train_02 =len(d2)
        item_num=0
        
        
        while (i<len_train_01 and j<len_train_02):
            distance_questions = word2vec_model.wv.wmdistance(d1[i], d2[j])
            #f.write(str(distance_questions))
            distance_list.append(distance_questions)
            i=i+1
            j=j+1
            item_num = item_num + 1
            if (item_num%100000 == 0):
                print ("Done so far {} ".format(item_num) )
        
        training_df['is_duplicate_01']=distance_list
        return training_df
        #return distance_list


# In[31]:


distance_calculation(training_error1_cf,training_error2_cf)


# In[32]:


#training_df.head()


# In[33]:


#feature_df[:10]


# In[34]:



item_num =0
for i, row in training_df['is_duplicate_01'].iteritems():
    if row == 0.000000:
        training_df.set_value(i,'is_duplicate_01',1)
    elif (row >= 0.28 and row <= 0.72):
        training_df.set_value(i,'is_duplicate_01',1)
    else:
        training_df.set_value(i,'is_duplicate_01',0)
    item_num = item_num + 1
    if (item_num%100000 == 0):
        print ("Done so far {} ".format(item_num) )
        
   


# In[35]:


#training_df.head()


# In[36]:


#training_df['is_duplicate_01'][:20]
#training_df['is_duplicate_01']=training_df[int(training_df['is_duplicate_01'])]
training_df['is_duplicate_01'] = training_df['is_duplicate_01'].astype(int)


# In[37]:


#training_df.head()


# In[38]:


#len(training_df)


# In[39]:


from sklearn.metrics import confusion_matrix,accuracy_score

y_actu = feature_df
y_pred = training_df['is_duplicate_01']
#cf_ma = confusion_matrix(y_actu, y_pred)
a_score=accuracy_score(y_actu, y_pred)


# In[40]:


cf_ma = confusion_matrix(y_actu, y_pred)
cf_ma


# In[1]:


#cf_ma
a_score


# In[42]:


#accuracy_score =67.30 when dimension=100
#chaging to 300 now


# In[43]:


def avg_length_corpus(training_error1_cf):
    
    min_length=0
    max_length=len(training_error1_cf[0])
    
    for i in range(0,len(training_error1_cf)):
        if len(training_error1_cf[i]) <=min_length :
            min_length= len(training_error1_cf[i])
        
    
    for i in range(1,len(training_error1_cf)):
        if len(training_error1_cf[i]) > max_length:
            max_length= len(training_error1_cf[i])
        
    
    avg_length= (min_length + max_length)//2
    
    
    return min_length,max_length,avg_length


# In[44]:


avg_question1_training_set= avg_length_corpus(training_error1_cf)


# In[45]:


avg_question2_training_set= avg_length_corpus(training_error2_cf)


# In[46]:


print(avg_question1_training_set)
print(avg_question2_training_set)


# In[47]:


def len_trainging_questions_array(training_error1_cf):
    questions_length_array =[]
    for i in training_error1_cf:
        questions_length_array.append(len(i))
        
    return questions_length_array
    


# In[48]:


question_1_length =len_trainging_questions_array(training_error1_cf)


# In[49]:


question_2_length =len_trainging_questions_array(training_error2_cf)


# In[50]:



import plotly.plotly as py
import plotly.graph_objs as go


trace = go.Box(
    y= question_2_length,
    name='Mean & SD',
    marker=dict(
        color='rgb(10, 140, 208)',
    ),
    boxmean='sd'
)
data = [trace]
#py.iplot(data)
plotly.offline.plot(data)


# In[51]:


def duplicate_entries(input_array):
    
    final_result=[]
    new_array=[]
    unseen=set()
    count=0
    for i in input_array:
        for j in i:
                 if j not in unseen:
                    unseen.add(j)
                    final_result.append(model[j])
                    #final_result.append(j)
        if len(final_result) >9:
            final_result=final_result[0:9]
        else:
            array_length=len(final_result)
            while array_length <9:
                final_result.append(np.zeros(200))
                array_length=array_length +1
                    
        new_array.append(final_result)
        final_result=[]
            
    
    return new_array
    
            


# In[52]:


vector_training_set_question1 =duplicate_entries(training_error1_cf)


# In[53]:


#print(type(vector_training_set_question1))


# In[54]:


vector_training_set_question2 =duplicate_entries(training_error2_cf)


            
        


# In[55]:


#vector_training_set_question2


# In[56]:


a = np.array(vector_training_set_question1)
b = np.array(vector_training_set_question2)




# In[57]:


result=np.column_stack((a,b))


# In[58]:


#len(result)


# In[59]:


#in_dim = 3600


# In[60]:


y=feature_df


# In[61]:


X=result


# In[62]:


len(X)



# In[63]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=42)


# In[64]:


#KNN


# In[65]:


len(X_test)


# In[66]:


'''
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.regularizers import L1L2
from keras.utils import np_utils
'''


# In[67]:


in_dim = 3600


# In[68]:


y=feature_df


# In[69]:


X=result


# In[2]:


from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=3)  
classifier.fit(X_train, y_train)


# In[ ]:


y_pred = classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))


# In[70]:


#from sklearn.model_selection import train_test_split


# In[71]:


'''
model = Sequential()
model.add(Dense(2, activation='relu'))
model.add(Dense(2, activation='softmax',input_dim=3600) )
'''


# In[1]:




#model


# In[73]:


#model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])


# In[ ]:


#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=22)


# In[ ]:


#len(X_test)


# In[ ]:


#X_train= X_train.reshape(323432, in_dim)


# In[ ]:


#X_train = X_train.astype('float32')


# In[ ]:


#X_test = X_test.reshape(80858, in_dim)


# In[ ]:


#X_test = X_test.astype('float32')


# In[ ]:


#Y_train = np_utils.to_categorical(y_train)
#Y_test = np_utils.to_categorical(y_test)


# In[ ]:



#history  == model.fit(X_train,  Y_train,  validation_split =0.10,epochs=25,batch_size=64)


# In[ ]:


#score = model.evaluate(X_test, Y_test,verbose=0)


# In[ ]:


#print('Test score:', score[0])
#print('Test accuracy:', score[1])


# In[ ]:




#print('Test score:', score[0])
#print('Test accuracy:', score[1])


# In[ ]:


#print(history.history.keys())


# In[ ]:


'''
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''


# In[ ]:


'''
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''


# In[ ]:


#predictions = model.predict(X_test)
#print('First prediction:', predictions[0])


# In[ ]:


#KNN

