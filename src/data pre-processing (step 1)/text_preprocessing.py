import re
import random
import numpy as np
import csv
list_of_stop_words = string_list = [
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves", "he", "him",
    "his", "himself", "she", "her", "hers", "herself", "it", "its",
    "itself", "they", "them", "their", "theirs", "themselves", "what",
    "which", "who", "whom", "this", "that", "these", "those", "am",
    "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "having", "do", "does", "did", "doing", "a", "an", "the",
    "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "s", "t",
    "can", "will", "just", "don", "should", "now"
]

def tokenize(lines, bol):
    for index in range(0,len(lines)):
        lines[index] = re.sub(r'[!"#$%&()*+/.:;\\<=>@\[\]^`{|}~\t\n]', '', lines[index])
        lines[index] = lines[index].split()
        lines[index] = [lines[index], bol]
    return lines

def save_to_file(file_name, list_of_values):
    list_of_values = list(zip(*list_of_values))[0]
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        for i in list_of_values:
            writer.writerow(i)
    # np.savetxt(file_name, list(zip(*list_of_values))[0], delimiter=',', fmt='%s')
    print("Saved file: ", file_name)

def save_label_file(file_name, list_of_values):
    label_list = list(zip(*list_of_values))[1]
    np.savetxt(file_name, ["pos" if x else "neg" for x in label_list ] , delimiter=',', fmt='%s')

def main():
    """Implement your assignment solution here"""
    positive_tokens = []
    with open('./data/raw_data/pos.txt') as f:
        lines = f.readlines()
        positive_tokens = tokenize(lines, True)
        f.close()
    
    negative_tokens = []
    with open("./data/raw_data/neg.txt") as f:
        lines = f.readlines()
        negative_tokens = tokenize(lines, False)
        f.close()

    dataset_with_stop_words = positive_tokens + negative_tokens

    random.shuffle(dataset_with_stop_words)
    n = len(dataset_with_stop_words)
    slice_index_a = int(0.8 * n)
    slice_index_b = int(0.9 * n)

    train_data_with_stop_words = dataset_with_stop_words[:slice_index_a]
    val_data_with_stop_words = dataset_with_stop_words[slice_index_a: slice_index_b]
    test_data_with_stop_words = dataset_with_stop_words[slice_index_b : ]

    save_to_file("./data/train_val_test/out.csv", dataset_with_stop_words)
    save_to_file("./data/train_val_test/train.csv", train_data_with_stop_words)
    save_to_file("./data/train_val_test/val.csv", val_data_with_stop_words)
    save_to_file("./data/train_val_test/test.csv", test_data_with_stop_words)

    save_label_file("./data/train_val_test/out.label.csv", dataset_with_stop_words)
    save_label_file("./data/train_val_test/train.label.csv", train_data_with_stop_words)
    save_label_file("./data/train_val_test/test.label.csv", test_data_with_stop_words)
    save_label_file("./data/train_val_test/val.label.csv", val_data_with_stop_words)    

    dataset_without_stop_words = dataset_with_stop_words

    for i in range(0, len(dataset_without_stop_words)):
        dataset_without_stop_words[i][0] = [x for x in dataset_without_stop_words[i][0] if x.lower() not in list_of_stop_words]


    train_data_without_stop_words = dataset_without_stop_words[:slice_index_a]
    val_data_without_stop_words = dataset_without_stop_words[slice_index_a: slice_index_b]
    test_data_without_stop_words = dataset_without_stop_words[slice_index_b : ]

    print("Total length: ", n)
    print("Train length :", len(train_data_without_stop_words))
    print("Val data : ", len(val_data_without_stop_words))
    print("test data : ", len(test_data_without_stop_words))

    save_to_file("./data/train_val_test/out_ns.csv", dataset_without_stop_words)
    save_to_file("./data/train_val_test/train_ns.csv", train_data_without_stop_words)
    save_to_file("./data/train_val_test/val_ns.csv", val_data_without_stop_words)
    save_to_file("./data/train_val_test/test_ns.csv", test_data_without_stop_words)

    # save_label_file("./data/train_val_test/out_ns.label.csv", dataset_without_stop_words)
    # save_label_file("./data/train_val_test/train_ns.label.csv", train_data_without_stop_words)
    # save_label_file("./data/train_val_test/val_ns.label.csv", val_data_without_stop_words)
    # save_label_file("./data/train_val_test/test_ns.label.csv", test_data_without_stop_words)


if __name__ == "__main__":
    main()
