import sentencepiece
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import glob
import itertools
import json
import spacy
import pytextrank
import en_core_web_sm
import re
import sys
from deepmultilingualpunctuation import PunctuationModel


# punctuation model
model_p = PunctuationModel()
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def get_filenames(files):
    """ returns file names"""
    file_names = []
    for file in files:
        file_names.append(glob.glob(file))
    file_names = list(itertools.chain.from_iterable(file_names))
    return file_names

def make_summary_f(text, max_length):
    """ use the facebook model to get a summary"""
    sum_list = summarizer(text, min_length=20, max_length = max_length, do_sample=False)
    string = next(iter(sum_list[0].items()))[1]
    return string

def punct_model(text):
    """punctuate the text"""
    return(model_p.restore_punctuation(text))

def text_rank(text):
    """ use text rank to rank the phrases"""
    nlp = spacy.load('en_core_web_md')
    nlp.add_pipe('textrank', last=True)
    doc = nlp(text)
    str_arr = []
    for string in doc._.textrank.summary(limit_phrases=5, limit_sentences=1):
        str_arr.append(string)
    return str_arr[0].text

def model(text, run_text_rank, fb_length):
    """ run_text_rank is a bool, true if text rank should be
        used in the model. Set length for make_summary_f"""
    if run_text_rank:
        temp =  text_rank(text)
        return make_summary_f(temp,fb_length)
    
    return make_summary_f(text, fb_length)

def short_min(text):
    """
    cut text to first punctuation mark
    """
    symbols = [".",";"]
    arr = []
    i = 0
    counter = 0
    for char in text:
        if char in symbols:
            counter+=1
            arr.append((char, i))
        i+=1
        
    min_val = sys.maxsize
    for tup in arr:
        if min_val>tup[1]:
            min_val = tup[1]
    
    #if you have a good sentence of the right size return it
    return text[0:min_val]

def check_str(text,l,u):
    """ give lower and upper bounds and return true or false
        if the text is within the range"""
    if text == 0:
        return False
    elif len(text)<=u and len(text)>=l:
        return True
    else:
        return False

def shorten_string(text):
    """
    cut text to closest punctuation mark to 80 characters
    """
    symbols = [".",";"]
    arr = []
    i = 0
    counter = 0
    #get all locations of the punctuations
    for char in text:
        if char in symbols:
            counter+=1
            arr.append((char, i))
        i+=1
        
    min_val = sys.maxsize
    for tup in arr:
        if min_val>tup[1]:
            min_val = tup[1]
    
    i = 0
    up_bound = 0
    low_bound = 0
    # get punctuations closest to character 90
    while i<len(arr):
        if up_bound !=0 and low_bound !=0:
            break
        elif arr[i][1]>=90:
            up_bound = arr[i][1]
            low_bound = arr[i-1][1]
        i+=1
    #print(arr)
    #print(up_bound, low_bound)
    
    difu = abs(up_bound-90)
    difl = abs(low_bound-90)
    diful = up_bound-low_bound
    
    # if the upper and lower bounds are close
    if up_bound !=0:
        if diful<=5 or difu<=difl or difl<35:
            short_string = text[0:up_bound]
            return short_string
        elif difu>=difl:
            short_string = text[0:low_bound]
            return short_string
    
    return text[0:min_val]
    
def check_stops(text):
    """return how many full stops in the text"""
    symbols = ["."]
    counter = 0
    for char in text:
        if char == ".":
            counter+=1
    return counter

def similar(len1, len2):
        if abs(len1-len2)<=3:
            return True
        return False

def cut(a, l, u, grp):
    # check if the model gives a suitable summary
    
    # shorten the text to the first punctuation and see if it meets the criteria    
    temp_f = short_min(a)

    # shorten the text to the punctutation mark closest to 80 characters and see if it
    # meets the criteria
    temp_e = shorten_string(a)
    
    # add punctuation to the SHORTENED TEXT to FIRST punctuation. 
    # shorten it again to the FIRST punctuation and check if it meets the criteria
    punct_temp_f = punct_model(temp_f)
    temp_fpf = short_min(punct_temp_f)

    # add punctuation to the SHORTENED TEXT to FIRST punctuation.
    # shorten it again to the 80TH punctuation and check if it meets the criteria
    temp_fpe = shorten_string(punct_temp_f)

    # add punctuation to the SHORTENED TEXT to 80TH punctuation.
    # shorten it again to the FIRST punctuation and check if it meets the criteria
    punct_temp_e = punct_model(temp_e)
    temp_epf = short_min(punct_temp_e)

    # add punctuation to the SHORTHENED TEXT to 80TH punctuation.
    # shorten it again to the 80TH punctuation and check if it meets the criteria
    temp_epe = shorten_string(punct_temp_e)

    # add punctuaction to the TEXT and shorten it to the FIRST punctuation
    punct_temp_t = punct_model(a)
    temp_tpf = short_min(punct_temp_t)

    # add punctuation to the TEXT and shorten it to the 80TH punctuation
    temp_tpe = shorten_string(punct_temp_t)

    summaries = []
    lengths = []
    summaries.extend([a, temp_f, temp_e, temp_fpf, temp_fpe, temp_epf, temp_epe, temp_tpf, temp_tpe])
    lengths.extend([len(a), len(temp_f), len(temp_e), len(temp_fpf), len(temp_fpe), len(temp_epf)
                    , len(temp_epe), len(temp_tpf), len(temp_tpe)])
    
    summaries.sort()
    lengths.sort()
    i = len(summaries)-1
    while i>0:
        if not similar(lengths[i],lengths[8]) and check_str(summaries[i],l,u):
            return str(grp + ": " + summaries[i])
        elif not similar(lengths[i],lengths[8]) or check_str(summaries[i],l,u):
            return str(grp + ": " + summaries[i])
        i-=1
    
    return str(grp + ": " + summaries[8])
    

def main():
    path = ["Json files/*.json"]
    json_files = get_filenames(path)
    i=0
    with open('jsons.txt', 'a') as g:
        while i < len(json_files):
            f = open(json_files[i])
            dicts = json.load(f)
            for vals in dicts.values():
                for val in vals.items():
                    l = 40
                    u = 150
                    if val[0] == "PROBLEM DESCRIPTION" or val[0] == "TARGET CONDITION" or val[0] == "CURRENT CONDITION" or val[0] == "ROOT CAUSE ANALYSIS" or val[0] == "COUNTERMEASURES" or val[0] == "EFFECT CONFIRMATION" or val[0] == "FOLLOW UP ACTION":
                        g.write("\n")
                        a = model(val[1], False, 53)
                        if len(a)<5:
                            g.write(val[1])
                        else:
                            # uncut string
                            # g.write(str(val[0] + ": " + a))
                            # cut string
                            g.write(cut(a,l,u,val[0]))
            i+=1

if __name__=="__main__":
    main()



