import argparse

from gensim.models import Word2Vec

def main(args):

    # Path to text file:
    with open(args.word_doc, 'r') as file:
        word_list = [line.strip() for line in file]

    # Loading the word2vec model:
    w2v = Word2Vec.load("a3/data/word2vec")

    # Top 20 most similar words:
    for word in word_list:
        similar_words_list = w2v.wv.most_similar(word, topn=20)
        print(f"Similar words for '{word}':")
        for word, similarity in similar_words_list:
            print(word, f"{similarity:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Similar Words')
    parser.add_argument('word_doc', type=str, help='Path to text document')
    args = parser.parse_args()
    main(args)