# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 15:18:42 2016

@author: Laura Drummer
"""
from py2neo import Graph, Node, Relationship
import config

graph = Graph(config.neo_graph)


print("Updating Neo4J")

def topic_word_graph(topic_words):
    for n in range(config.NMF_settings['num_topics']):
       topic = Node("Topic", name="Topic "+str(n))
       for i in range(config.NMF_settings['num_topic_words']):
           word = Node("Word", name=topic_words[n][i])
           graph.merge(Relationship(topic, "CONSISTS_OF", word))



def art_pairs_graph(art_network):
    for author in art_network.keys():
        if len(author) <= 30:
            

#def write_cypher()

 for author in (list(random.sample(art_network.keys(),5))): #sample - to test out cypher
        subset.append(author)#for testing only
        a_code = cy_formatter(author)
        declared_aliases.append(a_code)
        print('MERGE ('+a_code+':Person {name: "'+author+'"})',file=cof)
        for recipient in list(art_network[author].keys()):
            #for testing only
            r_code = cy_formatter(recipient)
            subset.append(recipient)
            if r_code not in declared_aliases:
                declared_aliases.append(r_code)
                print('MERGE ('+r_code+':Person {name: "'+recipient+'"})',file=cof)
            avg_message = art_network[author][recipient]['avg_message_len']
            reciprocated = art_network[author][recipient]['reciprocated']
            weight = art_network[author][recipient]['weight']
            r_topics = (['t'+number_formatter(n) for n in (art_network[author]
                        [recipient]['topics'].argsort()[-3:][::-1]).tolist()])
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
                m_code = "m"+cy_formatter(message)
                if m_code not in declared_aliases:
                    declared_aliases.append(m_code)
                    subject = cy_quote_cleaner(message_dict[message]['subject'])
                    print('MERGE ('+m_code+':Message {name: "'+message
                      +'", subject: "'+subject+'"})',file=cof)
                    m_topics = ['t'+number_formatter(n) for n in (
                (message_dict[message]['topics']
                .argsort()[-3:][::-1]).tolist())]
                    for topic in m_topics:
                        print('MERGE ('+m_code+')-[:BELONGS_TO]->('+topic+')'
                          , file=cof)
                print('MERGE ('+a_code+')-[:SENT]->('+m_code+')',file=cof)
                print('MERGE ('+r_code+')<-[:TO]-('+m_code+')',file=cof)