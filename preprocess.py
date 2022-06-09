import spacy


class document_preprocess:
    '''
    Input: Parameters for adding preprocessing operations to the pipeline 
    Description: Class containing functions for preprocessing and n-grams
    '''
    def __init__(self,lemmatize=True, stop_words=True, singleton=True, valid_word=True, custom_stop_words=[]):
        self.lemmatize=lemmatize
        self.stop_words=stop_words
        self.singleton=singleton
        self.valid_word=valid_word
        self.custom_stop_words=custom_stop_words
        self.nlp = spacy.load("en_core_sci_lg")

    def filter_word(self, text):
        ''' 
        Input: Text document
        Description: This function inputs a paerticular document text, creates it to an nlp object and then performs the following 
                     preprocessing operations-
                     1. Lemmatization
                     2. Stop words removal
                     3. Singleton removal
                     4. Custom stop words removal
                     5. Valid word check
        Output: Preprocessed text
        '''
        
        filtered_sentence=[]
        doc = self.nlp(str(text).lower())
        
        for word in doc:
            filters=[]
            if self.lemmatize:
                word=word.vocab[word.lemma_]

            filters.append(True if word.is_alpha else False)

            # dont append word if it is a stop word
            if self.stop_words:
                filters.append(False if word.is_stop else True)

            # dont append word if its length is 1
            if self.singleton:
                filters.append(False if len(word.text)==1 else True)

            # dont append word if it belongs to custom stop word
            if len(self.custom_stop_words)>0:
                filters.append(False if word.text in self.custom_stop_words else True)

            # If there is a valid word vector
            if self.valid_word:
                filters.append(False if word.vector.sum()==0 else True)

            if all(filters):
                filtered_sentence.append(word.text)

        return filtered_sentence
    
    def make_ngrams(self,s,n):
        ''' 
        Input: String
        Description: Create N-grams of a string
        Output: n-grams
        '''
        ngrams=[]
        s=self.nlp(s)
        for n in range(1,n+1):
            ngrams.extend([s[i:i+n] for i in range(len(s)-n+1)])
        return ngrams