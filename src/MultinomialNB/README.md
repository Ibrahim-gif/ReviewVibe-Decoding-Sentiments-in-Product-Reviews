# Add execution instructions here

The Train, Test, Validation data are read from

For each of the unigram, bigram, and unigram + bigram we evaluate the model and predict to get the accuracies 

Next, we create an instance of CountVectorizer() and pass the parameters (1,1) for unigaram, (1,2) for uni+bi gram model and (2,2) for bigram model

The following table describes the results of the modle with stop words and without stop words

Stopwords removed | text features | Accuracy (test set)
yes                 unigrams            80.33%
yes                 bigrams             78.85%
yes                 unigrams+bigrams    82.04%
no                  unigrams            80.56%
no                  bigrams             82.48%
no                  unigrams+bigrams    83.19%

The modle is first traind on the train set and validation set is uesd to find the best hyper parameters

The the best hyperparameter values are used to train the modle and the test dataset is used to evaluate the 

The performance of including or excluding stopwords can vary depending on the specific task and dataset. Stopwords are commonly occurring words in a language (e.g., "the", "is", "and") that are often removed during text preprocessing. By removing stopwords, we aim to reduce noise and focus on the more informative words. However, in certain tasks like sentiment analysis, stopwords can carry valuable sentiment-related information. Words like "not" or "no" can significantly affect the sentiment of a sentence. Therefore, removing stopwords in such cases may result in the loss of crucial sentiment indicators. On the other hand, for tasks that rely more on topical or keyword-based analysis, removing stopwords might improve performance by reducing the dimensionality of the data and eliminating irrelevant noise. Ultimately, the decision to include or exclude stopwords should be based on the specific requirements and characteristics of the task and dataset.

The Unigram + bigram model performes the best 
The bigrma model performs the second best
The Unigram model performance is the least

The bigram model performs slightly better than the unigram model due to its ability to capture contextual information. In the case of language modeling or text prediction, the previous word in a bigram provides important context that aids in predicting the probability of the next word. This context allows the model to better understand the dependencies and relationships between words. By incorporating bigrams, the model gains more specific and fine-grained information, leading to improved performance. Additionally, the combination of unigrams and bigrams in the unigram + bigram model provides a larger dataset, allowing the model to learn from a broader range of patterns and further enhance its predictive capabilities.

