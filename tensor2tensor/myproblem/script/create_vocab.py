from collections import Counter

from tensor2tensor.data_generators import text_encoder
from tqdm import tqdm

_DATA_PATH = "./data/train.source"
_STORE_PATH = './data/vocab.headline_gen.'

VOCAB_LEN = 0

def create_vocab(min_count=None):
    counter = Counter()
    print("opening data file...")
    data_file = open(_DATA_PATH)
    senlist = data_file.readlines()
    wordlist = []
    for sen in tqdm(senlist):
        wordlist.extend(sen.split())
    print("counting words...")
    counter.update(wordlist)
    vocab = set()
    min_count = 0 if not min_count else min_count
    print("geting vocab tokens...")
    for w,c in tqdm(counter.items()):
        if c > min_count:
            vocab.add(w)
    VOCAB_LEN = len(vocab)
    print("storing vocab file... ")
    with open(_STORE_PATH + str(VOCAB_LEN) + '.TOKEN', 'w') as f:
        f.write(text_encoder.PAD + '\n')
        f.write(text_encoder.EOS + '\n')
        f.write('<UNK>' + '\n')
        for w in tqdm(vocab):
            f.write(w+'\n')
            
if __name__ == '__main__':

    create_vocab(50)
