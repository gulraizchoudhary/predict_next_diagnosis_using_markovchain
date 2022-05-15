# -*- coding: utf-8 -*-
"""
Created on Sat May 14 18:56:22 2022
Predict the next ICD-10 code based on the dataset with markov chains.
@author: G. I. Choudhary
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from ConfusionMatrix import getCM, printStat


# Markov chain stored as adjacency list.
markov = {}

def update_markov(current : str, next : str) -> None:
    """Add item to the markov.
    Args:
        current (three digit ICD-10 codes is str): Input ICD-10 code.
        next (ICD-10 code as str): Output ICD-10 code.
    """

    # Add the input ICD-10 code to the lexicon if it in there yet.
    if current not in markov:
        markov.update({current: {next: 1} })
        return

    # Retrieve the probabilties of the input ICD-10 code.
    options = markov[current]

    # Check if the output ICD-10 codes that is in the propability list.
    if next not in options:
        options.update({next : 1})
    else:
        options.update({next : options[next] + 1})

    # Update the markov
    markov[current] = options

def normalize() -> None:
    """normalize the frequencies to a 0-1 float"""
    for code, transition in markov.items():
        transition = dict((key, value / sum(transition.values())) for key, value in transition.items())
        markov[code] = transition
    

def predict(code : str) -> str:
    """Attempt to predict the next ICD-10 code in the markov chain.
    Args:
        ICD-10 code (str): Last ICD-10 known code from patient history.
    Returns:
        str: Next ICD-10 code.
        None: current ICD-10 is not in the markov chain.
    """
    if code not in markov:
        return None

    options = markov[code]
    return np.random.choice(list(options.keys()), p=list(options.values()))
    

def train_markov(train):
    """update the markov chain using train"""
    
    for line in train:
        # Update markov chain.
        code = line.strip().split(' ')
        for i in range(len(code) - 1):
            update_markov(code[i], code[i+1])

def predict_next(test):
    """Predict the next ICD-10 code from a given"""
    pred = []
    gt = []
    for line in test:
        codes = line.strip().split(' ')
        t_head, t_tail = codes[:len(codes)//2], codes[len(codes)//2:]
    
        # Select the last ICD-10 code.
        p_code = predict(t_head[-1])
        
        #book keeping the total number of the classes
        gt.append(t_tail[0])
        gt.append(p_code)
        
        pred.append((str(p_code),t_head[-1]))
        
    
    return pred, len(set(gt))


def load_dataset():
    """1- Load the dataset from the file
       2- Split dataset into train and test
    """
    data = tuple(open("dataset.txt", 'r'))
        
    train, test = train_test_split(data,test_size=0.2)
    
    return train, test


if __name__ == '__main__':
    #Split data into train and test
    train, test = load_dataset()
    
    #update the markov model
    train_markov(train)
    
    #Normalize the markov model
    normalize()
    
    try:
      predicted, classes  =  predict_next(test)
      cm = getCM(predicted, classes)
      printStat(cm)
    except (KeyboardInterrupt, EOFError):
        pass