# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 14:49:14 2016

@author: Laura Drummer
"""
from datetime import datetime
from config import mailSettings, er
import uuid


def subject_line_parser(subject):
    """Parses subj and returns stats on replies, forwards, and subj content

    Keyword arguments:
    subject -- Raw subject line parsed from email, may or may not contain
               message prefixes ("re" or "fwd")
    """
    # Create a counter for total 'reply' and 'forward' prefixes.
    rep = 0
    fwd = 0
    split_sub = subject.split(":")
    # Assume anything to the right of the last ':' is the subject
    subject = split_sub[-1].strip()
    prefixes = split_sub[:-1]
    # If the "prefix" is not 're' or 'fw' assume it is part of the subject and
    # reattach to the subject field, otherwise +1 the relevant counter.
    for prefix in prefixes:
        if prefix.strip() == 're':
            rep += 1
        elif prefix.strip() == 'fw':
            fwd += 1
        else:
            subject = prefix.lstrip() + ':' + subject
    # Return a tuple of counters, and subject line.
    return(rep, fwd, subject)


def field_cleaner(field):
    """Removes 'bad' characters from strings in mail header may perform ER

    Removes bad characters <,>,/,[mailto],and datetime stamps - optonally
    performs entity resolution (using hard coded text file - very simple)

    Keyword arguments:
    field -- Raw field parsed from email, may or may not contain bad chars
    """

    if '<' in field and '>' in field:
        name, field = field.split('<', maxsplit=1)
        field = field.strip('>')
        field = field.replace('/', '_')
    # Remove date time stamp from field if exists.
    if ' on ' in field and (' pm ' in field or ' am ' in field):
        field, date = field.split(' on ', 1)
    if '[mailto:' in field:
        field, trash = field.split(' [', 1)
    if '"' in field:
        field = field.replace('"', '')
    # Optionally perform entity resolution on fields, for testing only.
    if er['entity_resolution']:
        with open(er['path']) as erd:
            er_data = erd.readlines()
        for line in er_data:
            if field == line.rstrip().split(',')[0]:
                field = line.rstrip().split(',')[1]
                break
    return(field)


def email_parser(file_name, filtered_subs):
    """Extracts all metadata fields from raw email, creates dictionary

    Keyword arguments:
    file_name -- full file path to email file for parsing
    filtered_subs -- list of subject lines to ignore (for testing only)
    """
    email_dict = {}
    # Specific to Enron dataset - grabs all data to the right of 'maildir'
    local_folders = file_name.split('maildir')[-1].strip()
    email_dict['user'] = local_folders.split('\\')[1]
    # Grabs the sub folders in a given users box (ex: 'sent' , 'important')
    email_dict['sub_folder'] = "__".join(local_folders.split('\\')[2:-1])
    with open(file_name) as e:
        e_data = [line.rstrip().lower() for line in e.readlines()]
    header = True
    previous_field = ''
    index = 0
    for line in e_data:
        if header:
            index += 1
            try:
                field, data = line.split(':', maxsplit=1)
                email_dict[field] = data.lstrip()
            # If there is not ':', assume the header field has multiple lines.
            except ValueError:
                email_dict[previous_field] += " " + line
            # Once the field 'x-filename' appears, assume start of body.
            if field == 'x-filename':
                header = False
        else:
            email_dict['body'] = " ".join(e_data[index:])
        previous_field = field
    email_dict['body'] = email_dict['body'].replace('\t', ' ')
    # Tabs are messing up the tokenization.
    return(email_dict)


def calc_weight(method, total_recipients):
    """Calcs relative weight of a relationship using msg type, # of recipients

    Keyword arguments:
    method -- 'to', 'cc', or 'bcc'
    total_recipients -- number of recipients
    """

    if method == 'to':
        # If directly emailed, disperse weight between all on 'to' line.
        try:
            return(1 / total_recipients)
        except ZeroDivisionError:
            return(0)
    elif method == 'cc':
        # If recipient is only cc degrade it according to value in config file
        try:
            return(1 / total_recipients * mailSettings['cc_wt'])
        except ZeroDivisionError:
            return(0)
    elif method == 'bcc':
        # If recipient is only bcc degrade it according value in config file
        try:
            return(1 / total_recipients * mailSettings['bcc_wt'])
        except ZeroDivisionError:
            return(0)


def email_cleaner(email_dict):
    """Cleans and normalizes all fields extracted from the email_parser func.

    Keyword arguments:
    email_dict -- data structure returned from email_parser function, contains
                  parsed fields and message body
    """

    # Parsing out different types of recipients.
    rec_fields = ['to', 'cc', 'bcc']
    total_recs = 0
    for rec in rec_fields:
        try:
            email_dict[rec] = [field_cleaner(email.strip()) for email
                               in email_dict[rec].split(',')]
        except KeyError:
            email_dict[rec] = []
        total_recs += len(email_dict[rec])
    email_dict['total_recs'] = total_recs
    email_dict['from'] = field_cleaner(email_dict['from'])
    # Standardizing Dates.
    try:
        date_string = email_dict['date'].split('-')[0].strip()
        email_dict['date'] = datetime.strptime(date_string,
                                               '%a, %d %b %Y %H:%M:%S')
    except KeyError:
        email_dict['date'] = None

    # Parses Subject.
    try:
        email_dict['subject_parsed'] = subject_line_parser(
            email_dict['subject'])
    except KeyError:
        email_dict['subject'] = ''
        email_dict['subject_parsed'] = (0, 0, '')

    # Tokenizes body and performs word count.
    email_dict['token_body'] = [word for word
                                in email_dict['body'].strip().split(' ')
                                if word != '']
    email_dict['word_count'] = len(email_dict['token_body'])
    # possibly add other measures here

    # Generates Email Dictionary - Main key is an 'author-recipient pairs'
    email_dict['ar_pairs'] = {'to': {
        'pairs': [email_dict['from'] + "___" + rec for
                  rec in email_dict['to']],
        'weight': calc_weight(
            'to', len(email_dict['to']))
    },

        'cc': {
        'pairs': [email_dict['from'] + "___" + rec for
                  rec in email_dict['cc']],
        'weight': calc_weight(
            'cc', len(email_dict['cc']))
    },

        'bcc': {'pairs': [email_dict['from'] + "___" + rec for
                          rec in email_dict['bcc']],
                'weight': calc_weight(
            'bcc', len(email_dict['bcc']))
    }
    }
    return(str(uuid.uuid4()), email_dict)
