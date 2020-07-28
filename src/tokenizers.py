class WordPunctEmojiTokenizer:
    
    def __init__(self, min_word_freq=5, max_word_freq=0.85, parse_emoji=False):
        
        """
        Attributes:
        --------------
        min_word_freq: float
            minimal number of documents wich are contains word
            
        max_word_freq: float
            maximal fraction of documents wich are contains word
            
        Functions:
        --------------
        
        fit:
        word_to_tok:
        tok_to_word:
        
        Notes:
        --------------
        
        """

        from collections import defaultdict
        import numpy as np
        
        self.total_samples = 0
        self.tok_unk = 0
        self.punct = np.array(['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',',
                               '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', 
                               ']', '^', '_', '`', '{', '|', '}', '~'])
        
        self.min_word_freq = min_word_freq
        self.max_word_freq = max_word_freq
        self.parse_emoji = parse_emoji
                
        self.word_total_freq = defaultdict(int) # here we store total word frequency
        self.word_sentence_freq = defaultdict(int) # here we store word frequency in each sentence
        
        self.word_tok_dict = defaultdict(int) # dict to convert word to token
        self.tok_word_dict = defaultdict(str) # dict to convert token to word
        
        
    def tokenize_sentence(self, sentence):
 
        tokens_final = []
        for token in wordpunct_tokenize(sentence):
            
            if re.findall('[^\w\d\s]', token): # this part check for stacked punctuation 
                if np.all(np.isin(np.array(list(token)),  self.punct)):
                    tokens_final.extend(list(token))
                else:
                    tokens_final.append(token)
            else:
                tokens_final.append(token)

        return tokens_final
    
    
    def tokenize_emoji_sentence(self, sentence):
        # Tokenize sentence with emojies in correct way
        from nltk import wordpunct_tokenize
        
        tokens = []
        for tok in wordpunct_tokenize(sentence):
            if emoji.get_emoji_regexp().search(tok):
                tokens.extend([i for i in list(tok) if i != ''])
                
            elif re.findall('[^\w\d\s]', tok):
                if np.all(np.isin(np.array(list(tok)),  self.punct)):
                    tokens.extend(list(tok))
                else:
                    tokens.append(tok)
            else:
                tokens.append(tok)

        return tokens
     
        
    def fit(self, corpus):
        
        """
        corpus: np.ndarray
            array of arrays of strings: array(array(str), array(str), ...)
            
        """
        
            
        import numpy as np
        from tqdm import tqdm
        from nltk import wordpunct_tokenize # our main tokenizer
        from collections import Counter
        
        
        if not isinstance(corpus, np.ndarray):
            raise TypeError(f'corpus has to be numpy.ndarray type, got {type(corpus)}')
        self.total_samples = corpus.shape[0]
        
        iter_ = 0
        for sentence in tqdm(corpus, position=0, leave=True):
            if isinstance(sentence, np.ndarray) or isinstance(sentence, list):
                sentence = sentence[0]
                
            if not isinstance(sentence, str):
                raise TypeError(f'Sentence must be string, found {type(sentence)} type on index {iter_}')
                        
                    
            # Emoji dealing with part
            if self.parse_emoji:
                tokens = self.tokenize_emoji_sentence(sentence)
                
            else:
                tokens = self.tokenize_sentence(sentence)
            
            tokens_freq = Counter(tokens)
            for key in tokens_freq.keys():
                self.word_total_freq[key] += tokens_freq[key] # total frequency 
                self.word_sentence_freq[key] += 1 # frequency in the sentence
                
            iter_ += 1
    
    
   
    
    def word_to_tok(self, corpus):
        
        """
        convert your corpus to tokens with settings
        """
        
            
        import numpy as np
        from collections import defaultdict
        from tqdm import tqdm
        from nltk import wordpunct_tokenize # our main tokenizer
        from collections import Counter
        
        self.word_tok_dict = {k: v for k, v in sorted(self.word_sentence_freq.items(), key=lambda item: item[1], reverse=True) 
                              if v >= self.min_word_freq and v <= int(self.max_word_freq * self.total_samples)}
        
        self.word_tok_dict = {k: idx + 3 for idx, k in enumerate(self.word_tok_dict.keys())}
        # (idx + 1) becouse 0 is unknown token
        # 1 - sos token, 2 - eos token
        
        self.tok_word_dict = {v: k for k, v in self.word_tok_dict.items()} # to reverse our tokenization
        
        def sentence_to_tokens(sentence):
            if isinstance(sentence, np.ndarray):
                sentence = sentence[0]

            if not isinstance(sentence, str):
                raise TypeError(f'sentence has to be str, got {type(sentence)}')
                
            if self.parse_emoji:
                tokens = self.tokenize_emoji_sentence(sentence)
            else:
                tokens = self.tokenize_sentence(sentence)
                
            encoded_sentence = []
            
            for tok in tokens:
                if tok in self.word_tok_dict.keys():
                    encoded_sentence.append(self.word_tok_dict[tok])
                else:
                    encoded_sentence.append(0)
                    
            return np.array(encoded_sentence)
        
        return np.array([sentence_to_tokens(i) for i in tqdm(corpus, position=0, leave=True)])
        
        
        
    def tok_to_word(self, tokens):
        
        """
        conver tokens on pretrained class to words back
        
        """
            
        import numpy as np
        from collections import defaultdict
        from tqdm import tqdm
        from nltk import wordpunct_tokenize # our main tokenizer
        from collections import Counter
        
        def tokens_to_sentence(toks):
            
            sentence = ''
            for tok in toks:
                if tok in self.tok_word_dict.keys():
                    sentence += self.tok_word_dict[tok]
                    sentence += ' '
                    
                elif tok == 0:
                    sentence += '<UNK>'
                    sentence += ' '
                    
                elif tok == 1:
                    sentence += ' '
                    
                elif tok == 2:
                    sentence += ' '
                    
                else:
                    sentence += '<UNK>'
                    sentence += ' '
                    
            return np.array(sentence)
        
        if isinstance(tokens[0], np.ndarray):
            return np.array([np.array(tokens_to_sentence(i)) for i in tqdm(tokens, position=0, leave=True)])
        
        else:
            return np.array(tokens_to_sentence(tokens))
    