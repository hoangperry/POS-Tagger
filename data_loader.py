import glob
import re
import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, data_dir='data/'):
        self.data_dir = data_dir
        self.processed_data = list()
        self.list_tag = list()
        self.list_word = dict()
        self.error_list = list()
        self.data_train = None
        self.data_test = None

    def load(self):
        for file_name in glob.glob(self.data_dir + '*/*.POS'):
            print('\r {}'.format(file_name), end='')
            self.preprocess_data_file(file_name)

        data_df = list()
        for senn in self.processed_data:
            sen_text = list()
            for tag in senn:
                sen_text += map(lambda x: x['word'], tag) if type(tag) is list else [tag['word']]
            data_df.append({
                'data': ' '.join(sen_text),
                'label': senn
            })
        data_df = pd.DataFrame(data_df)
        self.data_train, self.data_test = train_test_split(data_df, test_size=0.2, random_state=42)
        print('\nAll data is loaded!!')

    @staticmethod
    def clean_split_data(_file_name):
        sentences = list(map(
            lambda x: x.strip(), re.split('=======+', open(
                _file_name, mode='r', encoding='utf-8'
            ).read())[1:]
        ))
        ret = list()
        for line_data in sentences:
            if line_data.__len__() == 0:
                continue

            ret.append(list(map(
                lambda x: x.strip(),
                line_data.split('\n')
            )))
        return ret

    @staticmethod
    def strip_data_line(_line):
        return list(map(
            lambda x: x.split('/'),
            re.split(r' +', _line.strip(' '))
        ))

    def process_line(self, _line):
        ret = list()
        invalid_list = list()
        _is_sub = _line.startswith('[')
        _line = self.strip_data_line(_line[1:-2]) if _is_sub else self.strip_data_line(_line)
        for _tag in _line:
            if _tag.__len__() == 2:
                ret.append({
                    'word': _tag[0],
                    'tag': _tag[1]
                })
                if _tag[1] not in self.list_tag:
                    self.list_tag.append(_tag[1])
                if _tag[0] not in self.list_word:
                    self.list_word[_tag[0]] = [_tag[1]]
                else:
                    self.list_word[_tag[0]].append(_tag[1])
            else:
                invalid_list.append(_tag)

        return ret, _is_sub, invalid_list

    def preprocess_data_file(self, file_name, serial=True):
        for block in self.clean_split_data(file_name):
            _processed_block = list()
            for line in block:
                if line.__len__() == 0:
                    continue
                _return_set = self.process_line(line)
                if serial:
                    _processed_block += _return_set[0]
                else:
                    _processed_block += [_return_set[0]] if _return_set[1] else _return_set[0]
                self.error_list.append(_return_set[2])
            self.processed_data.append(_processed_block)
