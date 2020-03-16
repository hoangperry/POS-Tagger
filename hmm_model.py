from data_loader import DataLoader


class HMMBasic:
    def __init__(self):
        self.num_words_train = 0
        self.train_li_words = list()
        self.train_li_tags = list()
        self.dict2_tag_follow_tag_ = dict()
        self.dict2_word_tag = dict()
        self.dict_word_tag_baseline = dict()

    def load_data_train(self, data_train):
        for senn in data_train['label']:
            self.num_words_train += senn.__len__()
            for i in senn:
                self.train_li_words.append(i['word'])
                self.train_li_tags.append(i['tag'])

    def load_date_to_dict_tag(self):
        for i in range(self.num_words_train - 1):
            outer_key = self.train_li_tags[i]
            inner_key = self.train_li_tags[i + 1]

            self.dict2_tag_follow_tag_[outer_key] = self.dict2_tag_follow_tag_.get(outer_key, {})
            self.dict2_tag_follow_tag_[outer_key][inner_key] = self.dict2_tag_follow_tag_[outer_key].get(inner_key, 0)
            self.dict2_tag_follow_tag_[outer_key][inner_key] += 1

            outer_key = self.train_li_words[i]
            inner_key = self.train_li_tags[i]
            self.dict2_word_tag[outer_key] = self.dict2_word_tag.get(outer_key, {})
            self.dict2_word_tag[outer_key][inner_key] = self.dict2_word_tag[outer_key].get(inner_key, 0)
            self.dict2_word_tag[outer_key][inner_key] += 1

    def transform_probability(self):
        for key in self.dict2_tag_follow_tag_:
            di = self.dict2_tag_follow_tag_[key]
            s = sum(di.values())
            for innkey in di:
                di[innkey] /= s
            di = di.items()
            di = sorted(di, key=lambda x: x[0])
            self.dict2_tag_follow_tag_[key] = di

        for key in self.dict2_word_tag:
            di = self.dict2_word_tag[key]
            self.dict_word_tag_baseline[key] = max(di, key=di.get)
            s = sum(di.values())
            for innkey in di:
                di[innkey] /= s
            di = di.items()
            di = sorted(di, key=lambda x: x[0])
            self.dict2_word_tag[key] = di

    def train(self, data_train):
        self.load_data_train(data_train)
        self.load_date_to_dict_tag()

        self.dict2_tag_follow_tag_['.'] = self.dict2_tag_follow_tag_.get('.', {})
        new_fl_tag = self.dict2_tag_follow_tag_['.'].get(self.train_li_tags[0], 0)
        self.dict2_tag_follow_tag_['.'][self.train_li_tags[0]] = new_fl_tag
        self.dict2_tag_follow_tag_['.'][self.train_li_tags[0]] += 1

        last_index = self.num_words_train - 1

        outer_key = self.train_li_words[last_index]
        inner_key = self.train_li_tags[last_index]
        self.dict2_word_tag[outer_key] = self.dict2_word_tag.get(outer_key, {})
        self.dict2_word_tag[outer_key][inner_key] = self.dict2_word_tag[outer_key].get(inner_key, 0)
        self.dict2_word_tag[outer_key][inner_key] += 1
        self.transform_probability()

    def eval(self, data_test):
        sum_error = 0
        for idx in range(data_test.__len__()):
            test_li_words = list()
            test_li_tags = list()
            num_errors_baseline = 0
            output_li_baseline = list()
            for idx_word, word in enumerate(data_test.values[idx][1]):
                test_li_words.append(word['word'])
                test_li_tags.append(word['tag'])
                predict_tag = self.dict_word_tag_baseline.get(word['word'], '')
                if predict_tag == '':
                    output_li_baseline.append('NNP')
                else:
                    output_li_baseline.append(predict_tag)

                if predict_tag != test_li_tags[idx_word]:
                    num_errors_baseline += 1
            sum_error += num_errors_baseline / data_test.values[idx][1].__len__()
        return 1 - (sum_error / data_test.__len__())
