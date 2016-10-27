# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 07:39:08 2016

@author: Laura Drummer
"""
from datetime import datetime
import pandas as pd
import os
import uuid
import networkx as nx
import numpy as np
import scipy.sparse.linalg.eigen.arpack as eigenerror#error handling

project_folder = "C:\\Users\\Laura Drummer\\projects\\socialbee\\socialbee\\"
stop_words = project_folder+"custom_stop_words.txt"
#punct = project_folder+"custom_punct.txt"

with open(stop_words) as swf:
    enron_words = [word.rstrip() for word in swf.readlines()]

#with open(punct) as pf:
#    custom_punct = punct.read()
          
def subject_line_parser(subject):
    rep=0
    fwd=0
    split_sub = subject.split(":")
    subject = split_sub[-1].strip()
    prefixes = split_sub[:-1]
    for prefix in prefixes:
        if prefix.strip() == 're':
            rep+=1
        elif prefix.strip() =='fw':
            fwd+=1
        else:
            subject = prefix.lstrip()+':'+subject
    return(rep,fwd,subject)

def calc_weight(method, total_recipients):
    #function for caluclating relative weight message on relationship
    if method == 'to':
        #if directly emailed disperse weight between all on 'to' line        
        try:
            return(1/total_recipients)
        except ZeroDivisionError:
            return(0)
    else:
        #if recipient it only cc or bcc, degrade by 25%        
        try:
            return(1/total_recipients*.75)
        except ZeroDivisionError:
            return(0)

def field_cleaner(field):
    email = field
    if '<' in field and '>' in field:
        name, email = email.split('<')
        email = email.strip('>')
    if '/' in field:
        email = email.replace('/','_')
    if ' on ' in email and (' pm ' in email or ' am ' in email):
        email, date = email.split(' on ',1)
    if '[mailto:' in email:
        email, trash = email.split(' [',1)
    
    if '"' in field:
        email = email.replace('"','')
    return(email)
        

def email_parser(file_name):
    email_dict={}
    local_folders = file_name.split('maildir')[-1].strip()
    email_dict['user'] = local_folders.split('\\')[1]
    email_dict['sub_folder']="__".join(local_folders.split('\\')[2:-1])
    with open(file_name) as e:
        
        e_data = [line.rstrip().lower() for line in e.readlines()]
    header=True
    previous_field=''
    index=0
    for line in e_data:
        if header:
            index+=1            
            try:            
                field, data = line.split(':',maxsplit=1)
                email_dict[field]=data.lstrip()
            except ValueError:
                email_dict[previous_field]+=" "+line
            if field == 'x-filename':
                header=False
        else:
            email_dict['body'] = " ".join(e_data[index:])
        previous_field = field
    email_dict['body'] = email_dict['body'].replace('\t',' ') 
    #tabs are messing up the tokenization
    return(email_dict)

def email_cleaner(email_dict):                
    
    #parsing out different types of recipients    
    rec_fields = ['to','cc','bcc']
    total_recs =0    
    for rec in rec_fields:
        try:
            email_dict[rec] =[field_cleaner(email.strip()) for email 
            in email_dict[rec].split(',')]
        except KeyError:
            email_dict[rec] =[]
        total_recs += len(email_dict[rec])
    email_dict['total_recs'] = total_recs
    email_dict['from'] = field_cleaner(email_dict['from'])
    #standarizing dates    
    try:
        date_string = email_dict['date'].split('-')[0].strip()
        email_dict['date'] = datetime.strptime(date_string, 
                                               '%a, %d %b %Y %H:%M:%S')
    except KeyError:
        email_dict['date'] = None           
    
    #parsing subject
    try:
        email_dict['subject_parsed'] = subject_line_parser(
                                                      email_dict['subject'])
    except KeyError:
        email_dict['subject'] = ''
        email_dict['subject_parsed'] = (0,0,'')
    
    #message characterization
    email_dict['token_body'] = [word for word 
                                      in email_dict['body'].strip().split(' ')
                                      if word != '']
    email_dict['word_count'] = len(email_dict['token_body'])
    #possibly add other measures here
    
    #do this separately after ER?    
    email_dict['ar_pairs'] = {'to': {
                                    'pairs': [email_dict['from']+"___"+rec for 
                                                      rec in email_dict['to']],
                                    'weight': calc_weight(
                                                   'to',len(email_dict['to']))
                                    },
                              
                              'cc': {
                                    'pairs': [email_dict['from']+"___"+rec for 
                                                     rec in email_dict['cc']],
                                    'weight': calc_weight(
                                                  'cc',len(email_dict['cc']))
                                    },

                              'bcc':{
                                    'pairs' : [email_dict['from']+"___"+rec for
                                                    rec in email_dict['bcc']],
                                    'weight' :calc_weight(
                                                  'bcc',len(email_dict['bcc']))
                                    }                
                             }
    
    return(str(uuid.uuid4()), email_dict)
    
def dict_flattener(email_info):
    uid = email_info[0]
    email_dict = email_info[1]
    flat_list=[]    
    for key in email_dict['ar_pairs'].keys():
        for ar_pair in email_dict['ar_pairs'][key]['pairs']:
            flat_dict={}
            flat_dict['message_id'] = uid
            try:
                flat_dict['from'], flat_dict['to']= ar_pair.split('___')
                flat_dict['weight'] = email_dict['ar_pairs'][key]['weight']
            except ValueError:
                return(key)
            flat_dict['to_type'] = key
            flat_dict['ar_pair'] = ar_pair
            flat_list.append(flat_dict)
    return(flat_list)
    
def caluclate_network_attributes(Graph):
    all_nodes = Graph.nodes()
    graph_df = pd.Series(all_nodes).to_frame(name="Node")
    degree = nx.degree(Graph)
    degree_cen = nx.degree_centrality(Graph)
    in_degree = nx.in_degree_centrality(Graph)
    out_degree = nx.out_degree_centrality(Graph)
    closeness = nx.closeness_centrality(Graph)
    between = nx.betweenness_centrality(Graph)
    try:
        eigen = nx.eigenvector_centrality_numpy(Graph)
        graph_df['Eigenvector'] = pd.Series([eigen[node] 
                                            for node in all_nodes])
    except eigenerror.ArpackNoConvergence:
        #if no eigenvector can be caluclated, set all to zero
        graph_df['Eigenvector'] = pd.Series([0 for n in all_nodes])
    graph_df['Degree'] = pd.Series([degree[node] for node in all_nodes])
    graph_df['DegreeCentrality'] = pd.Series([degree_cen[node] 
                                              for node in all_nodes])
    graph_df['InDegree'] = pd.Series([in_degree[node] 
                                      for node in all_nodes])
    graph_df['OutDegree'] = pd.Series([out_degree[node] 
                                       for node in all_nodes])
    graph_df['Closeness'] = pd.Series([closeness[node] 
                                       for node in all_nodes])
    graph_df['Betweeness'] = pd.Series([between[node] 
                                        for node in all_nodes])
    
    return(graph_df)

def number_formatter(integer):
    #makes numbers cypher freindly
    if integer < 10:
        return('0'+str(integer))
    else:
        return(str(integer))

def cy_formatter(email_address):
    #formats email addresses for cypher statements
    bad_chars=[".","-","@","'"]
    for char in bad_chars:
        email_address = email_address.replace(char,"_")
    if email_address[0].isdigit():
        email_address = "e_"+email_address
    return(email_address)

def cy_quote_cleaner(string):
   string = string.replace('"',"'")
   return(string)

def make_a_attribute_table(attribute):
    
    node_df = pd.Series(node_info['ALL']['Node']).to_frame(name = 'Node')
    node_df['Global'] = node_info['ALL'][attribute]
    for n in (range(num_topics)):
        topic_title = 'Topic '+str(n)
        node_df[topic_title] = pd.Series().to_frame(name =topic_title)
#        for 
    return(node_df)


print("creating full file paths")
root_dir = 'F:\\Data\\enron_mail_20150507\\maildir\\allen-p'

message_dict={}
comms_list = []
art_network={}
print("parsing emails")
for folder, subs, files in os.walk(root_dir):
    for filename in files:
        f_info = email_cleaner(email_parser(folder+'\\'+filename))
        message_dict[f_info[0]] = f_info[1]
        extracted_info = dict_flattener(f_info)    
        comms_list+=extracted_info
        for rel in extracted_info:
            a,r,w = rel['from'],rel['to'],rel['weight']
            if a not in art_network.keys():
                art_network[a]={}
            if r not in art_network[a].keys():
                art_network[a][r]={'weight':0,
                                   'topics':np.zeros(20),
                                   'm_ids':[],
                                   'total_words':0,
                                   'avg_message_len':0,
                                   'reciprocated':0.0}
            art_network[a][r]['weight']+=w 
            #tracking relative weight of relationship
            if f_info[0] not in art_network[a][r]['m_ids']:
                art_network[a][r]['m_ids'].append(f_info[0]) 
                #list of all relevant messages
            art_network[a][r]['total_words'] += (message_dict[f_info[0]]
                                                 ['word_count'])
        
for a in art_network.keys():
    for r in art_network[a].keys():
        art_network[a][r]['avg_message_len'] = (
                                               art_network[a][r]['total_words']
                                                /
                                               len(art_network[a][r]['m_ids'])
                                               )
        try:                                       
            art_network[a][r]['reciprocated'] = (
                                                len(art_network[a][r]['m_ids'])
                                                /
                                                len(art_network[r][a]['m_ids'])
                                                )
        except KeyError:
            art_network[a][r]['reciprocated'] = 0
                                               
      
print("flattening dictionary")
ebunch = []
for a in art_network.keys():
    for r in art_network[a].keys():
        if art_network[a][r]['weight'] ==0:
            print(".")
        ebunch.append((a,r,art_network[a][r]['weight']))
            
commsDF = pd.DataFrame(comms_list)

print("creating network graph")
commsDiG = nx.from_pandas_dataframe(commsDF, source='from', target='to',
                                    create_using =nx.DiGraph())
w_comms = nx.DiGraph()
w_comms.add_weighted_edges_from(ebunch)

###############################################################################
############### Topic Modelling #######################################


#import sklearn.feature_extraction.text as text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import decomposition


num_topics = 20
num_topic_words = 20
print("creating document term matrix")
tfidf = TfidfVectorizer(stop_words=enron_words, lowercase=True, 
                        strip_accents="unicode", use_idf=True, 
                        norm="l2", min_df = 5, max_df = .8)

fixed_message_id= list(message_dict.keys())

A = tfidf.fit_transform([message_dict[key]['body'].encode('utf-8') 
                                for key in fixed_message_id])
num_terms = len(tfidf.vocabulary_)
terms = [""] * num_terms
for term in tfidf.vocabulary_.keys():
    terms[ tfidf.vocabulary_[term] ] = term
print("Caluclating {} topics.".format(num_topic_words))
model = decomposition.NMF(init="nndsvd", n_components=num_topics, max_iter=200)
W = model.fit_transform(A)
H = model.components_

topic_words = []
for topic in model.components_:
    #clf.components_ is an array of all the words mapped to each topic
    word_idx = np.argsort(topic)[::-1][0:num_topic_words]
    #grabs the last 20 words in each topic
    topic_words.append([terms[i] for i in word_idx])

###############################################################################
#############bringing it all together######################################
print("Bringing it all together.")

n=0
rec_fields = ['to','cc','bcc']

#enriching art_network with caluclated topics (summing all arays)
for m_id in fixed_message_id:
    message_dict[m_id]['topics']=W[n]
    for m_type in rec_fields:
        for r in message_dict[m_id][m_type]:
            art_network[message_dict[m_id]['from']][r]['topics']+=W[n]
    n+=1


#generating empty containers based on topic size
topic_based_network = {}
nx_topic_dictionary={}
for n in range(num_topics):
    topic_based_network[n]=[] #each topic gets a list of e_bunches
    nx_topic_dictionary[n]=nx.DiGraph() #each topic gets a Graph()
    
print("Building topic-based networks")
#creating ebunches then loading into Graph() I think I can do this in one step
node_to_topic_dict = {}
for author in art_network.keys():
    for recipient in art_network[author].keys():
        for n in range(num_topics):
            weight = (art_network[author][recipient]['topics'][n] *
                      art_network[author][recipient]['weight'])
            if author not in node_to_topic_dict.keys():
                node_to_topic_dict[author] = (art_network[author]
                                              [recipient]['topics'])
            else:
                node_to_topic_dict[author] += (art_network[author]
                                               [recipient]['topics'])
            if weight > .01:
                topic_based_network[n].append((author,recipient,weight))
    
for n in range(num_topics):
    nx_topic_dictionary[n].add_weighted_edges_from(topic_based_network[n])


#dictionary of DataFrames()
node_info={}
print("Calculating Global Network Attributes")
node_info['ALL'] = caluclate_network_attributes(w_comms)

for n in range(num_topics):
    print("Calculating attributes for Topic {}.".format(n))
    node_info['Topic '+str(n)] = caluclate_network_attributes(
                                                       nx_topic_dictionary[n])


print("predicting relationships")
topic_to_user_dict={}
#building topic: users dictionary
for user in node_to_topic_dict.keys():
    topics = (node_to_topic_dict[user].argsort()[-3:][::-1]).tolist()
    #indicies of 3 highest topics   
    topics.sort()    
    topics = ','.join([str(t) for t in topics])
    if topics not in topic_to_user_dict.keys():
        topic_to_user_dict[topics] = []
    topic_to_user_dict[topics].append(user)
may_know=[]

for topic in topic_to_user_dict:
    for user in topic_to_user_dict[topic]:
        if len(topic_to_user_dict[topic]) > 1:
            for n in range(len(topic_to_user_dict[topic])-1):
                if topic_to_user_dict[topic][n] != user:
                    isAuthor = (topic_to_user_dict[topic][n] 
                                 in w_comms[user].keys())
                    try:
                        isRecip = (user in w_comms[topic_to_user_dict[topic]
                                    [n]].keys())
                    except KeyError:
                        isRecip = False
                    if not (isAuthor and isRecip):
                        try:
                            path = nx.shortest_path(w_comms, 
                                             
                                             topic_to_user_dict[topic][n]
                                             )
                            may_know.append([user,topic_to_user_dict
                                            [topic][n],0.1])
                        except nx.NetworkXNoPath:
                            try:
                                path = nx.shortest_path(w_comms, 
                                             topic_to_user_dict[topic][n],
                                             user
                                             )
                                may_know.append([user,topic_to_user_dict
                                                 [topic][n],topic])
                            except nx.NetworkXNoPath:
                                pass
                            
inferred_net = w_comms

inferred_net.add_weighted_edges_from(may_know)                                    
                        
######################Cypher Update Statement##################################
cypher_out = "test.cypher"

print("Writing CYPHER Statements")

with open(cypher_out, "w") as cof:
    for n in range(num_topics):
        t_code = number_formatter(n)
        print('MERGE (t'+t_code+':Topic {name: "Topic '+t_code+'"})',file=cof)
        for i in range(num_topic_words):
            w_code = number_formatter(i)                
            comb_code=t_code+w_code
            print('MERGE (w'+comb_code+':Word {name: "'+
                   topic_words[n][i]+'"})',file=cof)
            print('MERGE (w'+comb_code+')<-[:CONSISTS_OF]-(t'+t_code+')'
                  , file=cof)
    subset=[]#for sample runs
    ac=0
    for author in ['phillip.allen@enron.com']: #sample - to test out cypher
        subset.append(author)#for testing only
        ac+=1 #code to make cypher aliases unique
        a_code = cy_formatter(author)+str(ac)
        print('MERGE ('+a_code+':Person {name: "'+author+'"})',file=cof)
        rc=0        
        for recipient in list(art_network[author].keys())[40:45]:
            subset.append(recipient)#for testing only
            rc+=1 #code to make cypher aliases unique
            r_code = cy_formatter(recipient)+str(ac)+str(rc)
            avg_message = art_network[author][recipient]['avg_message_len']
            reciprocated = art_network[author][recipient]['reciprocated']
            weight = art_network[author][recipient]['weight']
            r_topics = (['t'+number_formatter(n) for n in (art_network[author]
                        [recipient]['topics'].argsort()[-3:][::-1]).tolist()])
            print('MERGE ('+r_code+':Person {name: "'+recipient+'"})',file=cof)
            print('MERGE ('+a_code+r_code+':Relationship {name: "'+author
                  +"___"+recipient+'", avg_len:'+str(avg_message)+
                  ', reciprocated:'+str(reciprocated)+', weight:'+str(weight)
                  +'})',file=cof)
            print('MERGE ('+a_code+')-[:HAS]->('+a_code+r_code+')',file=cof)
            print('MERGE ('+r_code+')<-[:WITH]-('+a_code+r_code+')',file=cof)
            for topic in r_topics:
                    print('MERGE ('+a_code+r_code+')-[:BELONGS_TO]->('+topic+')'
                          , file=cof)
            for message in art_network[author][recipient]['m_ids']:
                m_code = "m"+cy_formatter(message)+"__"+str(ac)+str(rc)
                subject = cy_quote_cleaner(message_dict[message]['subject'])
                m_topics = ['t'+number_formatter(n) for n in (
                (message_dict[message]['topics']
                .argsort()[-3:][::-1]).tolist())]
                print('MERGE ('+m_code+':Message {name: "'+message
                      +'", subject: "'+subject+'"})',file=cof)
                print('MERGE ('+a_code+')-[:SENT]->('+m_code+')',file=cof)
                print('MERGE ('+r_code+')<-[:TO]-('+m_code+')',file=cof)
                for topic in m_topics:
                    print('MERGE ('+m_code+')-[:BELONGS_TO]->('+topic+')'
                          , file=cof)
        rel_c=0
        for relationship in may_know:
            rel_c+=1
            #adding some code to make it relevant to the sample
            #(take this out in production)
            if relationship[0] in subset or relationship [1] in subset:
                rel_code1 = cy_formatter(relationship[0])+str(rel_c)
                rel_code2 = cy_formatter(relationship[1])+str(rel_c)
                print('MERGE ('+rel_code1+':Person {name: "'+relationship[0]
                       +'"})',file=cof)
                print('MERGE ('+rel_code2+':Person {name: "'+relationship[1]
                       +'"})',file=cof)
                print('MERGE ('+rel_code1+')-[:MAY_KNOW]->('+rel_code2+')'
                      ,file=cof)
        
def generate_attribute_table(attribute):
    #there is almost certainly a fast way to do this with slicing
    attrib_table = []    
    for node in node_info['ALL'].Node:
        node_dict={}
        node_dict['GLOBAL'] = ([float(node_info['ALL'][node_info['ALL']
                               .Node==node][attribute])])
        for n in range(num_topics):
            topic_key = "Topic "+str(n)
            try:
                node_dict[topic_key] = ([float(node_info[topic_key][node_info
                                        [topic_key].Node==node][attribute])])
            except TypeError:
                node_dict[topic_key] =0
        attrib_table.append(node_dict)
    return(attrib_table)
    
    
for df in range(num_topics):
    topic_title = 'Topic '+str(df)
    t_df = node_info[topic_title]
#    try:
#        renee_eigen.append(float(t_df[t_df.Node=='stagecoachmama@hotmail.com'].InDegree))
#    except TypeError:
#        renee_eigen.append(0)
attributes = ['Eigenvector','DegreeCentrality','Degree','InDegree','OutDegree',
              'Closeness','Betweeness']        

print("creating Node reports")
Node_reports={}

for attribute in attributes:
    print("Writing report for "+attribute+"...")
    attrib_df = pd.DataFrame.from_dict(
                generate_attribute_table(attribute)
                ).transpose()
    attrib_df.columns = node_info['ALL'].Node
    Node_reports[attribute] = attrib_df

#node_df = make_a_attribute_table('Eigenvector')

import matplotlib.pyplot as plt
#plt.plot(range(num_topics+1),renee_eigen)
#plt.show()

########
#vectorizer = text.CountVectorizer(stop_words=enron_words,min_df=20)
#dtm = vectorizer.fit_transform([message_dict[key]['body'].encode('utf-8') 
#                                for key in message_dict.keys()]).toarray()

#vocab = np.array(vectorizer.get_feature_names())


#clf = decomposition.NMF(n_components=num_topics, random_state=1)

#topic_words = []
#for topic in clf.components_:
    #clf.components_ is an array of all the words mapped to each topic
#    word_idx = np.argsort(topic)[::-1][0:num_topic_words]
#    #grabs the last 20 words in each topic
#    topic_words.append([vocab[i] for i in word_idx])