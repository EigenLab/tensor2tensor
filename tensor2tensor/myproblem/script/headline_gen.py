import os

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators.text_problems import Text2TextProblem, VocabType
from tensor2tensor.data_generators.text_problems import text2text_txt_iterator, text2text_generate_encoded
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry

from tensor2tensor.utils import metrics

_ARTICLE_PATH = "./data/train.source"
_HEADLINE_PATH = "./data/train.target"

@registry.register_problem
class HeadlineGen(Text2TextProblem):

    @property
    def vocab_filename(self):
        return "vocab.headline_gen.266848.TOKEN"

    @property
    def vocab_type(self):
        return VocabType.TOKEN

    @property
    def is_generate_per_split(self):
        """
        true: generate samples for every split
        false: generate samples just for train
        """
        return False

    @property
    def dataset_splits(self):
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 9,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        del dataset_split

        article_file = open(_ARTICLE_PATH, 'r')
        headline_file = open(_HEADLINE_PATH, 'r')

        article_list = article_file.readlines()
        headline_list = headline_file.readlines()

        article_file.close()
        headline_file.close()

        for article, headline in zip(article_list, headline_list):
            article = article.strip()
            headline = headline.strip()
            yield{
                "inputs": article,
                "targets": headline
            }

    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        """
        override this function just for add "replace_oov = '<UNK>'
        """
        generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
        vocab_filename = os.path.join(data_dir, self.vocab_filename)
        encoder = text_encoder.TokenTextEncoder(vocab_filename, replace_oov='<UNK>')
        return text2text_generate_encoded(generator, encoder,
                                        has_inputs=self.has_inputs)


    