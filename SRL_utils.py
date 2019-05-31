"""
These functions are extracting different features out of an NAF-file to create
a semantic role labeling classifier. After the extraction, the features are
vectorised and classified with a svm. A predication on the gold labels is done
and a classification report and a confusion matrix are created.
"""

import pandas as pd
from sklearn import svm
from lxml import etree
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import spacy
from spacy.lang.nl.examples import sentences 
import gensim
nlp = spacy.load('nl_core_news_sm')
model = gensim.models.KeyedVectors.load_word2vec_format("word_embeddings_160/combined-160.txt", binary=False)

def open_xml(file, version = "combined"):#version = model for features, embedding or combined
    """
    opens the xml files and calls the feature extraction functions on each file,
    depending on the model
    """
    gold = list() #list for gold standard of semroles (Arg0, Arg1 etc.)
    features = list() #list for features (pos, lemma, morphofeat)
    with open ("DutchSRL_NAFDEP/"+file+".naf", "r") as xml_file:
        tree = etree.parse(xml_file)
        root = tree.getroot()
        if root.find("srl") is None:
            return [],[]        
               
        srl_feature_list = get_srl(root) #[{'Arg0': ['t28', 't29', 't30', 't31']}, ...]
        termdict = extract_term_features(root) # {'t5471': {'lemma': '.', 'pos': 'punct', 'morphofeat': 'LET()'}}
        advanced_termdict = get_token(root, termdict)      
        predicates_dict = get_predicate(root, advanced_termdict) # {t1:t3}, {t2:t3} which term belongs to which predicate
        feature_list, srl_list= get_roots(srl_feature_list, advanced_termdict)
        
        if version == "combined":
            advanced_feature_list = embed_word(feature_list, srl_list, version)
            final_features_list = connect_srl_features(advanced_feature_list, predicates_dict, advanced_termdict)
        elif version == "embedding":
            final_features_list = embed_word(feature_list, srl_list, version)
        elif version == "features":
            final_features_list = connect_srl_features(feature_list, predicates_dict, advanced_termdict)
 
        gold.extend(srl_list)
        features.extend(final_features_list) 
        
    return gold, features

    
def get_srl(root):      
    """
    extracts a list of dicts of the semantic roles and the relating target-ids
    [{'Arg1': ['t5470']}...]
    """
    srl = root.find("srl")
    srl_children = srl.getchildren()    
    srl_list = list()
    for predicate in srl_children:
        sem_roles = predicate.findall("role")
        for item in sem_roles:
            role = item.get("semRole")
            span = item.find("span")
            targets = span.findall("target")
            spanlist = list()
            for targ in targets:
                id_target = targ.get("id")
                spanlist.append(id_target)
            srl_list.append({role:spanlist})
    return(srl_list)
                
    
def extract_term_features(root):
    """
    extracts features (pos, lemma, morphofeat) from the terms
    """
    terms = root.find("terms")
    terms_children = terms.getchildren()
    term_dict = dict()
    for term in terms_children:
        term_id = term.get("id") #id of term
        termitems = term.items()
        feat_dict = dict()#dict with the features
        for item in termitems[1:4]:
            key, value = item
            feat_dict[key] = value
        span = term.find("span")
        for target in span:
            token_id = target.get("id")
            feat_dict["token_id"] = token_id
        term_dict[term_id] = feat_dict
    return term_dict

def get_token(root, term_dict):
    """
    extracts the token of each lemma and adds into a dict related to the id
    """
    text = root.find("text")
    text_children = text.getchildren()
    token_dict = dict()
    for word in text_children:     
        token_id = word.get("id") #e.g. w1234
        token = word.text.lower()
        token_dict[token_id]=token
    for key, value in term_dict.items():
        token_id = term_dict[key]["token_id"]
        word = token_dict[token_id]
        term_dict[key]["token_id"]=word
    return term_dict


def get_predicate(root, termdict):
    """
    extracts a dict with each term-id as a key and the term-id of 
    the connected predicate as a value. E.g. t5470 : t5468 means, the predicate
    of the term t5470 is the term t5468
    """  
    srl = root.find("srl")
    srl_children = srl.getchildren()
    target_list = list()
    pred_dict = dict() #returning a dict with term_id and predicate
    for predicate_el in srl_children:
        pred_id = predicate_el.get("id") #number of the predicate e.g. p1, p2 etc.
        pred_target_el = predicate_el.find("span/target")
        pred_target_id = pred_target_el.get("id") #id for the term of the predicate e.g. "t1234"     
        if predicate_el.find("role/span") is None:
            pred_dict[pred_target_id]=[pred_target_id]
            continue
        else:
            role_list = predicate_el.findall("role")
            for role in role_list:
                span = role.find("span")
                for target in span:
                    target_id = target.get("id")
                    pred_dict[target_id]=pred_target_id
    return pred_dict

def get_roots(srl_feature_list, termdict):
    """
    using a dependency parses over a chunk consisting of the aligned arguments 
    of a semrole. Extracts the root of the chunk and okenises the lemma
    """
    srl_list = list()
    feature_list = list()
    for srl in srl_feature_list: #srl = {arg1:[t523, t524, t525]}
        sentence = ""
        arg_targ = dict()
        for srl, list_item in srl.items(): #target = [t523, t524, t525]
            for target in list_item:
                lemma = (termdict[target]["lemma"])
                lemma = lemma.strip(".")
                arg_targ[lemma]=target
                sentence += lemma+" "
                
            doc = nlp(sentence)
            for token in doc:
                if token.dep_ == "ROOT":
                    for key, value in arg_targ.items():
                        if str(token) in str(key):
                            srl_list.append(srl)
                            feature_list.append({value:termdict[value]})
    return feature_list, srl_list
                

def embed_word(feature_list, srl_list, version = "combined"):
    """
    adding word embeddings as a feature on the token of the root
    """
    embed_list = list()
    for feature_dict in feature_list:
        array_dict = dict()
        if version == "combined":        
            for key, value in feature_dict.items():
                array_dict = {key:value}
                token = value["token_id"]
                if token not in model:
                    vector = model["unknown"]
                else:
                    vector = model[token]
                counter = 0
                for number in vector:
                    array_dict[key]["vec"+str(counter)] = number
                    counter += 1
            embed_list.append(array_dict)
            
        elif version == "embedding":
            for key, value in feature_dict.items():
                array_dict = dict()
                token = value["token_id"]
                if token not in model:
                    vector = model["unknown"]
                else:
                    vector = model[token]
                counter = 0
                for number in vector:
                    array_dict["vec"+str(counter)] = number
                    counter += 1
            embed_list.append(array_dict)
    return embed_list
                

def connect_srl_features(feature_list, predicates_dict, termdict):
    """
    takes SRL-list, list of the target ids and list of dicts of features.
    relates the features to the target ids and returns a list of the srl features,
    that correlates with the index of the srls
    """
    final_features_list = list() # list of the srl_features
    for feature_dict in feature_list: #adding the predicate lemma as feature
        for key, value in feature_dict.items():
            predicates_id = predicates_dict[key]
            newdict = value
            newdict["predicate"] = termdict[predicates_id]["lemma"]
            final_features_list.append(newdict)
    return(final_features_list)

def get_array_length(array):
    """ 
    Calculate the length of an array. 
    Use for sparse arrays instead of len(). Made by Andras Aponyi
    """
    counter = 0    
    for row in array:
        counter += 1        
    return counter

def run_svr(training_instances, testfeatures, goldlabel, testgold):
    """
    takes the instances and goldlabels (SRL), trains a svm classifier and 
    creates a confusion matrix
    """    
    svm.LinearSVR
    vec = DictVectorizer()
    lin_clf = svm.LinearSVC()
    training_length = get_array_length(training_instances)
    test_length = get_array_length(testfeatures)
    
    combined_instances = training_instances + testfeatures  
    combined_array = vec.fit_transform(combined_instances)  
    
    training_array = combined_array[0:training_length]
    test_array = combined_array[training_length:]
    lin_clf.fit(training_array, goldlabel) #training
    prediction = lin_clf.predict(test_array) #prediction
    
    cm = confusion_matrix(testgold, prediction) #confusion matrix
    cf = classification_report(testgold, prediction)
    print(cm)
    print(cf)
