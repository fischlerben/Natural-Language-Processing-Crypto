# Decrypting Cryptocurrencies through Natural Language Processing (NLP)

The Natural Language Processing techniques employed below were done so for the purposes of sentiment analysis; what are articles saying about different cryptocurrencies?  What is the current public sentiment surrounding these coins?  NewsApi and NLTK (Natural Language Tool Kit) library utilized.
![sentiment](https://www.marketmotive.com/market_motive/sentiment-analysis.jpg)

---

After setting my NewsAPI Key, the first step was to fetch Bitcoin & Ethereum news articles:
    
    from newsapi import NewsApiClient
    bitcoin_headlines = newsapi.get_everything(q="bitcoin", language="en", page_size=100, sort_by="relevancy")
    ethereum_headlines = newsapi.get_everything(q="ethereum", language="en", page_size=100, sort_by="relevancy")

Next was to create the Sentiment Scores DataFrame using a for-loop:

    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    
    bitcoin_sentiments = []
    for article in bitcoin_headlines["articles"]:
        try:
            text = article["content"]
            date = article["publishedAt"][:10]
            sentiment = analyzer.polarity_scores(text)
            compound = sentiment["compound"]
            pos = sentiment["pos"]
            neu = sentiment["neu"]
            neg = sentiment["neg"]
        
        bitcoin_sentiments.append({
            "text": text, "date": date, "compound": compound, "positive": pos, "negative": neg, "neutral": neu})
        
        except AttributeError:
            pass

The same process was performed for Ethersum.  After converting to a DataFrame using pd.DataFrame, we are left with the following DataFrame to work from:
![dataframe](/Screenshots/dataframe.png?raw=true)

A simple ".describe()" function shows us important numbers related to the Sentiment Score:
![describe](/Screenshots/describe.png?raw=true)

#### Ethereum had the highest mean positive score with a mean of .082, compared to bitcoin's mean positive score of .055.  
#### Ethereum had the highest compound score with a max of .883, compared to bitcoin's max compound score of .878.  
#### Ethereum had the highest positive score with a max of .347, compared to bitcoin's max positive score of .318.

---

## Tokenizing:

### Initial Imports:

    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer, PorterStemmer
    from string import punctuation
    import re
    lemmatizer = WordNetLemmatizer()
    
### Define Tokenizer Function that Tokenizes Sentences into root words:

    def tokenizer(text):
        """Tokenizes text."""
    
    sw = set(stopwords.words('english'))
    regex = re.compile("[^a-zA-Z ]")
    re_clean = regex.sub('', text)
    words = word_tokenize(re_clean)
    lem = [lemmatizer.lemmatize(word) for word in words]
    tokens = [word.lower() for word in lem if word.lower() not in sw.union(sw_addons)]
    return tokens
    
Now we can take a look at unique word counts:

    from collections import Counter
    from nltk import ngrams
    
    # Generate the Bitcoin N-grams where N=2
    bitcoin_text = ' '.join(bitcoin_df.text)
    bitcoin_processed = tokenizer(bitcoin_text)
    bitcoin_ngrams = Counter(ngrams(bitcoin_processed, n=2))
    
We can then use the imported Counter to count the frequency of words in the articles:

    Counter(tokens).most_common(N)

The top 3 most frequently used words in the Bitcoin news articles were "char" (98x), "Bitcoin" (84x) and "Reuters" (71x).

---

## Word Clouds:
Word clouds are an intuitive way to visualize the frequency of different words in a news article to quickly see which words were most prominently used.  Word clouds are also very easily generated with Python:

### Initial Imports:    
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-whitegrid')
    import matplotlib as mpl
    mpl.rcParams['figure.figsize'] = [20.0, 10.0]
    
### Generate Word Cloud:

    wc = WordCloud().generate(bitcoin_text)
    plt.imshow(wc)

The above code generates the following Bitcoin Word Cloud:
![word_cloud](/Screenshots/word_cloud.png?raw=true)

---

## Named Entity Recognition (NER):
NER generates visually-appealing text that makes it clear what words are important within the article, and to what "category" that word belongs to: is it an organization, a currency, a name, etc.

### Initial Imports:
    
    import spacy
    from spacy import displacy
    nlp = spacy.load('en_core_web_sm')
    
First, concatenate all Bitcoin/Ethereum text together using the ".join()" function.  Then, run NER processor on text, and render visualization:

    bitcoin_doc = nlp(bitcoin_text)
    displacy.render(bitcoin_doc, style='ent')
    
The above code generates the following Named Entity Recognition (this is only a sample of the output):
![ner](/Screenshots/ner.png?raw=true)