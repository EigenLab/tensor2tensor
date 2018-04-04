import tensorflow as tf
import numpy as np

_PATHS = [
    "/home/zhuyuhe/mydata/headline_gen/model/tensor2tensor/tensor2tensor/myproblem/data/headline_gen-train-00000-of-00009",
]

_VOCAB_PATH = "/home/zhuyuhe/mydata/headline_gen/model/tensor2tensor/tensor2tensor/myproblem/data/vocab.headline_gen.266848.TOKEN"

def word_id_map():
    vocab_file = open(_VOCAB_PATH)
    vocab_list = vocab_file.readlines()
    word2id = {w:i for i,w in enumerate(vocab_list)}
    id2word = {i:w for i,w in enumerate(vocab_list)}
    return word2id, id2word

if __name__ == '__main__':
    with tf.Session() as sess:
        example = tf.train.Example()
        record_iterator = tf.python_io.tf_record_iterator(path = _PATHS[0])
        i=0
        _, id2word = word_id_map()
        for record in record_iterator:
            example.ParseFromString(record)
            features = example.features.feature
            i += 1
            input_ids = features['inputs'].int64_list.value
            print("input length: {0}".format(len(input_ids)))
            target_ids = features['targets'].int64_list.value
            print("target length: {0}".format(len(target_ids)))
            word_article_list = [id2word[id].strip('\n') for id in input_ids]
            word_headline_list = [id2word[id].strip('\n') for id in target_ids]
            print("article:\n" + "".join(word_article_list))
            print("headline:\n" + "".join(word_headline_list))
            if i >= 3:
                break