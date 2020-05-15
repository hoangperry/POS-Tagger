import re
import pickle


def viterbi(_obs, _states, _s_pro, _t_pro, _e_pro):
    # init
    path = {
        s: []
        for s in _states
    }
    curr_pro = dict()
    for s in _states:
        curr_pro[s] = _s_pro[s] * _e_pro[s][_obs[0]]
    for i in range(1, len(_obs)):
        last_pro = curr_pro
        curr_pro = {}
        for curr_state in _states:
            max_pro, last_sta = max((
                    (
                        last_pro[last_state] * _t_pro[last_state][curr_state] * _e_pro[curr_state][_obs[i]],
                        last_state
                    )
                    for last_state in _states
                ), key=lambda x: x[0]
            )

            curr_pro[curr_state] = max_pro
            path[curr_state].append(last_sta)

    max_pro = -1
    max_path = None
    for s in _states:
        path[s].append(s)
        if curr_pro[s] > max_pro:
            max_path = path[s]
            max_pro = curr_pro[s]
    return max_path


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

    def load_data_to_dict_tag(self):
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
        self.load_data_to_dict_tag()

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

    def predict_raw_text(self, text):
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r' +', ' ', text)
        result = list()
        for idx_word, word in enumerate(text.split(' ')):
            predict_tag = self.dict_word_tag_baseline.get(
                word, ''
            )
            if predict_tag == '':
                result.append((word, 'NNP'))
            else:
                result.append((word, predict_tag))
        return result

    @staticmethod
    def load(model_path):
        model_ = pickle.load(open(model_path, 'rb'))
        return model_


if __name__ == '__main__':
    states = (
        'Healthy',
        'Fever'
    )
    observations = (
        'normal',
        'cold',
        'dizzy'
    )

    start_probability = {
        'Healthy': 0.6,
        'Fever': 0.4
    }

    transition_probability = {
        'Healthy': {'Healthy': 0.7, 'Fever': 0.3},
        'Fever': {'Healthy': 0.4, 'Fever': 0.6},
    }

    emission_probability = {
        'Healthy': {
            'normal': 0.5,
            'cold': 0.4,
            'dizzy': 0.1
        },
        'Fever': {
            'normal': 0.1,
            'cold': 0.3,
            'dizzy': 0.6
        },
    }

    obs = [
        'normal',
        'cold',
        'dizzy'
    ]

    resut_vtb = viterbi(
        obs,
        states,
        start_probability,
        transition_probability,
        emission_probability
    )
    print(resut_vtb)
