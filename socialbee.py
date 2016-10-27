# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 07:39:08 2016

@author: Laura Drummer
"""
import os
import networkx as nx
import numpy as np
import pandas as pd
import config
import mail_parse
import network_builder
import report_writer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import decomposition


with open(config.spam['path']) as spam:
    spam_email = [email.rstrip() for email in spam.readlines()]

with open(config.enrich['stop_words']) as swf:
    enron_words = [word.rstrip() for word in swf.readlines()]

with open(config.enrich['sub_filter']) as subf:
    filtered_subs = subf.readlines()
    filtered_subs = [line.rstrip().lower() for line in filtered_subs]


def number_formatter(integer):
    # makes numbers cypher freindly
    if integer < 10:
        return('0' + str(integer))
    else:
        return(str(integer))


def cy_formatter(email_address):
    # formats email addresses for cypher statements
    bad_chars = [".", "-", "@", "'"]
    for char in bad_chars:
        email_address = email_address.replace(char, "_")
    if email_address[0].isdigit():
        email_address = "e_" + email_address
    return(email_address)


def cy_quote_cleaner(string):
    string = string.replace('"', "'")
    return(string)

print("creating full file paths")
root_dir = config.data_path

message_dict = {}
comms_list = []
art_network = {}
print("parsing emails")
for folder, subs, files in os.walk(root_dir):
    for filename in files:
        uid, mail_info = mail_parse.email_cleaner(mail_parse.email_parser
                                                  (folder + '\\' +
                                                   filename, filtered_subs)
                                                  )
        if config.spam['spam_filter'] and (
            mail_info['from'] in spam_email or
            len(mail_info['from']) > 30
        ):
            pass
        else:
            message_dict[uid] = mail_info
            extracted_info = network_builder.dict_flattener(uid, mail_info)
            comms_list += extracted_info
            for rel in extracted_info:
                a, r, w = rel['from'], rel['to'], rel['weight']
                if a not in art_network.keys():
                    art_network[a] = {}
                if r not in art_network[a].keys():
                    art_network[a][r] = {'weight': 0,
                                         'topics': np.zeros(
                                             config.NMF_settings['num_topics']
                                         ),
                                         'm_ids': [],
                                         'total_words': 0,
                                         'avg_message_len': 0,
                                         'reciprocated': 0.0}
                art_network[a][r]['weight'] += w
                # tracking relative weight of relationship
                if uid not in art_network[a][r]['m_ids']:
                    art_network[a][r]['m_ids'].append(uid)
                    # list of all relevant messages
                art_network[a][r]['total_words'] += (message_dict[uid]
                                                     ['word_count'])

for a in art_network.keys():
    for r in art_network[a].keys():
        art_network[a][r]['avg_message_len'] = (
            art_network[a][r]['total_words'] /
            len(art_network[a][r]['m_ids'])
        )
        try:
            art_network[a][r]['reciprocated'] = (
                len(art_network[a][r]['m_ids']) /
                len(art_network[r][a]['m_ids'])
            )
        except KeyError:
            art_network[a][r]['reciprocated'] = 0


print("flattening dictionary")
ebunch = []
for a in art_network.keys():
    for r in art_network[a].keys():
        if art_network[a][r]['weight'] == 0:
            print(".")
        ebunch.append((a, r, art_network[a][r]['weight']))

commsDF = pd.DataFrame(comms_list)

print("creating network graph")
commsDiG = nx.from_pandas_dataframe(commsDF, source='from', target='to',
                                    create_using=nx.DiGraph())
w_comms = nx.DiGraph()
w_comms.add_weighted_edges_from(ebunch)

# ----------------------------------------------------------------------------
# ------------------------Topic Modelling-------------------------------------


# import sklearn.feature_extraction.text as text

print("creating document term matrix")
tfidf = TfidfVectorizer(stop_words=enron_words,
                        lowercase=config.tfidf_settings['lowercase'],
                        strip_accents=config.tfidf_settings['strip_accents'],
                        use_idf=config.tfidf_settings['use_idf'],
                        norm=config.tfidf_settings['norm'],
                        min_df=config.tfidf_settings['min_df'],
                        max_df=config.tfidf_settings['max_df'])

fixed_message_id = list(message_dict.keys())

A = tfidf.fit_transform([message_dict[key]['body'].encode('utf-8')
                         for key in fixed_message_id])
num_terms = len(tfidf.vocabulary_)
terms = [""] * num_terms
for term in tfidf.vocabulary_.keys():
    terms[tfidf.vocabulary_[term]] = term
print("Caluclating {} topics.".format(config.NMF_settings['num_topics']))
model = decomposition.NMF(init=config.NMF_settings['init'],
                          n_components=config.NMF_settings['num_topics'],
                          max_iter=config.NMF_settings['max_iter'])
W = model.fit_transform(A)
H = model.components_

topic_words = []
for topic in model.components_:
    # clf.components_ is an array of all the words mapped to each topic
    word_idx = np.argsort(
        topic)[::-1][0:config.NMF_settings['num_topic_words']]
    # grabs the last 20 words in each topic
    topic_words.append([terms[i] for i in word_idx])

# -------------------------------------------------------------------------
#------------Bringing it all together------------------------------------
print("Bringing it all together.")
topic_threshold = config.may_know_settings['topic_threshold']

n = 0

# enriching art_network with caluclated topics (summing all arays)
for m_id in fixed_message_id:  # for every single unique message
    # update the dictionary with the topic array
    message_dict[m_id]['topics'] = W[n]
    message_dict[m_id]['top_topics'] = ([n for n in (
                                        message_dict[m_id]['topics']
                                        .argsort()[(-1 * (topic_threshold)):]
                                        [::-1].tolist())if n > 0]
                                        )

    n += 1


for a in art_network.keys():
    for r in art_network[a].keys():
        art_network[a][r]['topics'] = (sum([message_dict[m_id]['topics']
                                            for m_id in
                                            art_network[a][r]['m_ids']])
                                       /
                                       len(art_network[a][r]['m_ids'])
                                       )
        art_network[a][r]['top_topics'] = ([n for n in (
                                            art_network[a][r]['topics']
                                            .argsort()[(-1 * (topic_threshold)
                                            ):][::-1]).tolist() if n > 0])

# generating empty containers based on topic size
topic_based_network = {}
nx_topic_dictionary = {}
for n in range(config.NMF_settings['num_topics']):
    topic_based_network[n] = []  # each topic gets a list of e_bunches
    nx_topic_dictionary[n] = nx.DiGraph()  # each topic gets a Graph()

print("Building topic-based networks")
# creating ebunches then loading into Graph() I think I can do this in one step
node_to_topic_dict = {}
for author in art_network.keys():
    for recipient in art_network[author].keys():
        for n in range(config.NMF_settings['num_topics']):
            topics_array = np.copy(art_network[author][recipient]['topics'])
            # copying fixed a problem where the array was being modified
            rel_weight = art_network[author][recipient]['weight']
            weight = (topics_array[n] * rel_weight)
            if author not in node_to_topic_dict.keys():
                node_to_topic_dict[author] = (topics_array)
            else:
                node_to_topic_dict[author] += (topics_array)
            if weight > config.tbn_threshold:
                topic_based_network[n].append((author, recipient, weight))

for n in range(config.NMF_settings['num_topics']):
    nx_topic_dictionary[n].add_weighted_edges_from(topic_based_network[n])


# dictionary of DataFrames()
node_info = {}
print("Calculating Global Network Attributes")
node_info['ALL'] = network_builder.caluclate_network_attributes(w_comms)

for n in range(config.NMF_settings['num_topics']):
    print("Calculating attributes for Topic {}.".format(n))
    node_info['Topic ' + str(n)] = network_builder.caluclate_network_attributes(
        nx_topic_dictionary[n])


print("predicting relationships")
topic_to_user_dict = {}
# building topic: users dictionary

for user in node_to_topic_dict.keys():
    topics = (node_to_topic_dict[user].argsort()[
              (-1 * (topic_threshold)):][::-1]).tolist()
    topics = [index for index in topics if node_to_topic_dict[user][index] > 0]
    if config.may_know_settings['sorted_topics']:
        topics.sort()
    topics = ','.join([str(t) for t in topics])
    if topics not in topic_to_user_dict.keys():
        topic_to_user_dict[topics] = []
    topic_to_user_dict[topics].append(user)
may_know = []
may_know_dict = []

for topic in topic_to_user_dict:
    for user in topic_to_user_dict[topic]:
        if (len(topic_to_user_dict[topic]) > 1 and 
        len(topic_to_user_dict[topic]) <= config.may_know_settings['max_len']):
            for n in range(len(topic_to_user_dict[topic]) - 1):
                if topic_to_user_dict[topic][n] != user:
                    isAuthor = (topic_to_user_dict[topic][
                                n] in w_comms[user].keys())
                    try:
                        isRecip = (user in w_comms[topic_to_user_dict[topic]
                                                   [n]].keys())
                    except KeyError:
                        isRecip = False
                    if not (isAuthor and isRecip):
                        may_know.append([user,
                                         topic_to_user_dict[topic][n],
                                         topic[:5]])
                        may_know_dict.append(
                            {"user1": user,
                            "user2": topic_to_user_dict[topic][n],
                            "topics": topic[:5]}
                            )
                        # try:
                        #    path = nx.shortest_path(w_comms,
                        #
                        #                     topic_to_user_dict[topic][n]
                        #                     )
                        #    may_know.append([user,topic_to_user_dict
                        #                    [topic][n],config.may_know_settings['rel_wt']])
                        # except nx.NetworkXNoPath:
                        #    try:
                        #        path = nx.shortest_path(w_comms,
                        #                     topic_to_user_dict[topic][n],
                        #                     user
                        #                     )
                        #        may_know.append([user,topic_to_user_dict
                        #                         [topic][n],topic])
                        #    except nx.NetworkXNoPath:
                        #        pass

inferred_net = w_comms.copy()

inferred_net.add_weighted_edges_from(may_know)

######################Cypher Update Statement##################################
cypher_out = "test.cypher"

print("Writing CYPHER Statements")
import random  # for subset testing

with open(cypher_out, "w") as cof:
    for n in range(config.NMF_settings['num_topics']):
        t_code = number_formatter(n)
        print('MERGE (t' + t_code +
              ':Topic {name: "Topic ' + t_code + '"})', file=cof)
        for i in range(config.NMF_settings['num_topic_words']):
            w_code = number_formatter(i)
            comb_code = t_code + w_code
            print('MERGE (w' + comb_code + ':Word {name: "' +
                  topic_words[n][i] + '"})', file=cof)
            print('MERGE (w' + comb_code +
                  ')<-[:CONSISTS_OF]-(t' + t_code + ')', file=cof)
    subset = []  # for sample runs
    declared_aliases = []
    for author in (list(random.sample(art_network.keys(), 5))):  # sample - to test out cypher
        subset.append(author)  # for testing only
        a_code = cy_formatter(author)
        declared_aliases.append(a_code)
        print(
            'MERGE (' + a_code + ':Person {name: "' + author + '"})', file=cof)
        for recipient in list(art_network[author].keys()):
            # for testing only
            r_code = cy_formatter(recipient)
            subset.append(recipient)
            if r_code not in declared_aliases:
                declared_aliases.append(r_code)
                print(
                    'MERGE (' + r_code + ':Person {name: "' + recipient + '"})', file=cof)
            avg_message = art_network[author][recipient]['avg_message_len']
            reciprocated = art_network[author][recipient]['reciprocated']
            weight = art_network[author][recipient]['weight']
            r_topics = (['t' + number_formatter(n) for n in (art_network[author]
                                                             [recipient]['topics'].argsort()[-3:][::-1]).tolist()])
            print('MERGE (' + a_code + r_code + ':Relationship {name: "' + author
                  + "___" + recipient + '", avg_len:' + str(avg_message) +
                  ', reciprocated:' + str(reciprocated) +
                  ', weight:' + str(weight)
                  + '})', file=cof)
            print(
                'MERGE (' + a_code + ')-[:HAS]->(' + a_code + r_code + ')', file=cof)
            print(
                'MERGE (' + r_code + ')<-[:WITH]-(' + a_code + r_code + ')', file=cof)
            for topic in r_topics:
                print('MERGE (' + a_code + r_code +
                      ')-[:BELONGS_TO]->(' + topic + ')', file=cof)
            for message in art_network[author][recipient]['m_ids']:
                m_code = "m" + cy_formatter(message)
                if m_code not in declared_aliases:
                    declared_aliases.append(m_code)
                    subject = cy_quote_cleaner(
                        message_dict[message]['subject'])
                    print('MERGE (' + m_code + ':Message {name: "' + message
                          + '", subject: "' + subject + '"})', file=cof)
                    m_topics = ['t' + number_formatter(n) for n in (
                        (message_dict[message]['topics']
                         .argsort()[-3:][::-1]).tolist())]
                    for topic in m_topics:
                        print(
                            'MERGE (' + m_code + ')-[:BELONGS_TO]->(' + topic + ')', file=cof)
                print(
                    'MERGE (' + a_code + ')-[:SENT]->(' + m_code + ')', file=cof)
                print(
                    'MERGE (' + r_code + ')<-[:TO]-(' + m_code + ')', file=cof)

        for relationship in may_know:
            # adding some code to make it relevant to the sample
            #(take this out in production)
            if (relationship[0]) in subset and (relationship[1] in subset):
                rel_code1 = cy_formatter(relationship[0])
                rel_code2 = cy_formatter(relationship[1])
                print('MERGE (' + rel_code1 +
                      ')-[:MAY_KNOW]->(' + rel_code2 + ')', file=cof)


def generate_attribute_table(attribute, node_info, num_topics):
    # there is almost certainly a fast way to do this with slicing
    attrib_table = []
    for node in config.interesting_users:
        node_dict = {}
        node_dict['GLOBAL'] = ([float(node_info['ALL'][node_info['ALL']
                                                       .Node == node][attribute])])
        for n in range(num_topics):
            topic_key = "Topic " + str(n)
            try:
                node_dict[topic_key] = (
                    [float(
                        node_info[topic_key][node_info]
                        [topic_key].Node == node[attribute])])
            except TypeError:
                node_dict[topic_key] = 0
        attrib_table.append(node_dict)
    return(attrib_table)


for df in range(config.NMF_settings['num_topics']):
    topic_title = 'Topic ' + str(df)
    t_df = node_info[topic_title]
#    try:
#        renee_eigen.append(float(t_df[t_df.Node=='stagecoachmama@hotmail.com'].InDegree))
#    except TypeError:
#        renee_eigen.append(0)
attributes = ['Eigenvector', 'DegreeCentrality', 'Degree', 'InDegree', 'OutDegree',
              'Closeness', 'Betweeness']

print("creating Node reports")


for attribute in attributes:
    print("Writing report for " + attribute + "...")
    attrib_df = pd.DataFrame.from_dict(
        generate_attribute_table(attribute, node_info,
                                 config.NMF_settings['num_topics'])
    ).transpose()
    attrib_df.columns = node_info['ALL'].Node
    node_info[attribute] = attrib_df


ar_topics_l = []
for a in art_network.keys():
    for r in art_network[a].keys():
        t_dict = {'ar_pair': a + ',' + r}
        n = 0
        for n in range(config.NMF_settings['num_topics']):
            t_dict['Topic ' + str(n)] = art_network[a][r]['topics'][n]
        ar_topics_l.append(t_dict)

topics_df = pd.DataFrame(ar_topics_l)

print("Printing Reports")

with open('ART_report.txt', 'w') as af:
    for n in range(config.NMF_settings['num_topics']):
        topic = 'Topic ' + str(n)
        print(topics_df.nlargest(5, topic)[['ar_pair', topic]], file=af)

print("{} new 'may know' links".format(len(may_know)))
print("{} total links".format(len(w_comms.edges())))
print(len(may_know) / len(w_comms.edges()))


for topic in nx_topic_dictionary.keys():
    report_writer.network_report(
        topic,
        nx_topic_dictionary[topic],
        config.interesting_users,
        "demo_reports\\Topic {} Report.txt".format(topic),
        node_info["Topic {}".format(topic)],
        topic_words[topic])
    nx.write_gexf(nx_topic_dictionary[topic],
                  "demo_reports\\Topic {}.gexf".format(topic))


report_writer.global_writer(
    topic_words, w_comms, nx_topic_dictionary, may_know_dict)
nx.write_gexf(w_comms, "demo_reports\\ALL.gexf")
