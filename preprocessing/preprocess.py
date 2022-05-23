import pandas as pd
import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
""" classes created to preprocess and create a machine learning model
on an inputted set of statements/text for similarity comparisons
"""
class Preprocess:
    def __init__(self, input_file):  # input file is a string filepath 
        type = input_file[-3:]
        if type == "json":  # pandas reads differently based on filetype
            self.df = pd.read_json(input_file, lines=True)  # in either case, load data as a data frame
        elif type == "csv":
            self.df = pd.read_csv(input_file)

    def tokenize(self, column, new_column):
        """ column: string column name for the column in self.df containing statements
        new_column: string new column name for the column in self.df storing tokenized statements
        """
        tokens = []
        for i in range(len(self.df)):  # go through every row of dataframe
            tokens.append(str(self.df[column][i]).lower())  # lowercase all text in each statement
        self.df[new_column] = tokens  # append lowercase statements
        self.df[new_column] = self.df[new_column].apply(nltk.word_tokenize)  # tokenize all statements in tokens column

    def remove_empty(self, column):
        """ removes any row of self.df that has an empty list
        in the inputted colum; sometimes nltk.word_tokenize returns
        an empty list
        column: string column name 
        """
        for i in range(len(self.df)):
            if not self.df[column][i]:  # if list of tokens is empty
                self.df.drop(index=i, inplace=True)  # remove that row
        self.df.reset_index(inplace=True)  # reset the index to remove gaps for missing rows
