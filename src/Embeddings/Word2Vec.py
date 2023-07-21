import gensim
import pandas as pd

def main():
    """Implement your assignment solution here"""
    
    #import the csv file
    with open("./data/raw_data/pos.txt") as f:
        lines = f.readlines()
    
    #convert to dataframe
    df = pd.Series(lines)

    with open("./data/raw_data/neg.txt") as f:
        lines = f.readlines()
    
    df._append(pd.Series(lines))

    #Tokenize and pre-process
    process_tok = df.apply(gensim.utils.simple_preprocess)
    # print(process_tok)
    model = gensim.models.Word2Vec(
        window = 5,
        min_count = 2
    )

    model.build_vocab(process_tok, progress_per = 1000)
    # print(model.epochs)
    model.train(process_tok, total_examples = model.corpus_count, epochs = model.epochs)
    model.save("./data/word2vec")
    print("The top most similar words to good are \n")
    for word, score in model.wv.most_similar("good", topn = 20):
        print(word, ":  ", score)

    print("--------------------------------")

    print("The top most similar words to bad are \n")
    for word, score in model.wv.most_similar("bad", topn = 20):
        print(word, ":  ", score)
    


if __name__ == "__main__":
    main()
