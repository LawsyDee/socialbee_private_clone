# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:10:59 2016

@author: Laura Drummer
"""

#For testing purposes only - eventually config file needs to be created
#that doesn't allow any user to access/change

#select the data to 
data_path = 'F:\\Data\\enron_mail_20150507\\maildir\\sample'
users_path = 'demo_users.txt'
neo_graph = "http://neo4j:password@localhost:7474/"
reports_folder = 'demo_reports\\'

spam = dict(
            spam_filter = True,
            path = 'demo_spam_list.txt'
            )
            
er = dict (
           entity_resolution = True,
           path = 'rough_entity_resolution.txt'
           )

#adjust enrichment files (stop words, ER-files, etc)
enrich = dict(
                       stop_words ='demo_stop_words.txt',
                       sub_filter = 'subject_filter.txt'
                   )



#turn these dials to adjust the way the topic modelling works
tfidf_settings = dict( 
                      lowercase=True, 
                      strip_accents="unicode", 
                      use_idf=True, 
                      norm="l2", 
                      min_df = 5, 
                      max_df = .75
                      )

NMF_settings = dict(
                    num_topic_words = 20,
                    num_topics = 20,
                    init="nndsvd", #Method used to initialize the procedure
                    max_iter=200
                    )

#topic based network settings

tbn_threshold = .01

may_know_settings = dict (
                            rel_wt = 100,
                            topic_threshold = 5, 
                            sorted_topics = True,
                            max_len = 10
                         )




mailSettings = dict(
                        cc_wt=.75,
                        bcc_wt = .75
                    )

interesting_users= [
					'matthew.lenhart@enron.com',
					'mike.grigsby@enron.com',
					'jay.reitmeyer@enron.com',
					'phillip.allen@enron.com',
					'jane.m.tholt@enron.com',
					'frank.ermis@enron.com',
					'keith.holst@enron.com',
					'tori.kuykendall@enron.com'
					]