from gensim.models import Word2Vec
import numpy as np
import csv
import tensorflow as tf
from tensorflow import keras

def read_csv(filename):
    temp = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            temp.append(row)
    return temp

def word_embed(x,y, word2vec_model):
    corp_embed = []
    for sent in x:
        sent_embed = []
        for word in sent:
            if word in word2vec_model.wv.key_to_index:
                word_embeding = word2vec_model.wv[word]
                sent_embed.append(word_embeding)
    
        #Append the sentence embedded vector to the corpus embedding
        corp_embed.append(sent_embed)

    target = [1 if i[0] == "pos" else 0 for i in y]
    return corp_embed, target

def padding(x):
    input_size = 512
    for i in range(0,len(x)):
        if len(x[i]) <= input_size:
            x[i] = np.pad(x[i], [0, input_size - len(x[i])])
        else:
            x[i] = x[i][0:input_size]
    return x
def main():
    """Implement your assignment solution here"""
    #Load the model
    Word2Vecmodel = Word2Vec.load("./data/word2vec_model")

    #Load the files
    X_train = read_csv('./data/train_val_test/train.csv')
    X_test = read_csv('./data/train_val_test/test.csv')
    y_train = read_csv('./data/train_val_test/train.label.csv')
    y_test = read_csv('./data/train_val_test/test.label.csv')
    X_val = read_csv('./data/train_val_test/val.csv')
    y_val = read_csv('./data/train_val_test/val.label.csv')

    X_train, y_train = word_embed(X_train, y_train, Word2Vecmodel)
    X_test, y_test = word_embed(X_test, y_test, Word2Vecmodel)
    X_val, y_val = word_embed(X_val, y_val, Word2Vecmodel)

    #padding the inputs
    X_train = padding(X_train)
    X_test = padding(X_test)
    X_val = padding(X_val)

    #np array
    y_train = np.array(y_train)
    X_train = np.array(X_train)
    y_test = np.array(y_test)
    X_test = np.array(X_test)
    y_val = np.array(y_val)
    X_val = np.array(X_val)

    print("X Train: ", X_train.shape)
    print("y Train", y_train.shape)
    print("X test", X_test.shape)
    print("y test", y_test.shape)
    print("X val", X_val.shape)
    print("y val ", y_val.shape)

    #Defining the Model
    activation_functions = ["relu", "sigmoid", "tanh"]
    l2_norm = [0.001, 0.01, 0.1]

    best_accuracy = 0
    best_activation_fun = None
    best_l2_val = None

    models_ = {}

    for fun in activation_functions:
        for norm in l2_norm:
            model = keras.Sequential([
                            keras.layers.Dense(128, activation=fun, input_shape=(512,), kernel_regularizer=keras.regularizers.l2(norm)),
                            keras.layers.Dropout(0.25),
                            keras.layers.Dense(2, activation='softmax')
                        ])

            # Compile the model
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            # Train the model
            batch_size = 128
            num_epochs = 5

            model.fit(np.array(X_train), np.array(y_train), batch_size=batch_size, epochs=num_epochs, validation_data=(np.array(X_val), np.array(y_val)))

            accuracy = model.evaluate(np.array(X_val), np.array(y_val))
            models_[fun+"_"+ str(norm)] = model
            print(f"Activation function: {fun}   l2_norm: {norm}  Accuracy: {accuracy[0]}")
            if(accuracy[0] > best_accuracy):
                best_accuracy = accuracy[0]
                best_activation_fun = fun
                best_l2_val = norm

    best_model = best_activation_fun + "_" + str(best_l2_val)
    # Evaluate the accuracy on the test set

    model = models_[best_model]
    accuracy = model.evaluate(X_test, y_test)
    print("Accuracy on test set is: ", accuracy[0] * 100)

if __name__ == "__main__":
    main()
