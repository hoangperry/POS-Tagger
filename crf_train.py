import pickle
from data_loader import DataLoader
import sklearn_crfsuite

mini_data = True


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def word2features(sent, i):
    word = sent[i][0]
    features = {
        'word[-4:]': word.lower()[-4:],
        'mention': word.startswith('@') and len(word) > 1,
        'hashtag': word.startswith('#') and len(word) > 1,
        'word.lower()': word.lower(),
        'number': is_number(word),
        'word[-3:]': word.lower()[-3:],
        'word[-2:]': word.lower()[-2:],
        'word[-1:]': word.lower()[-1:],
        'word.istitle()': word.istitle(),
        'word.isupper()': word.isupper(),
    }

    return features


def get_features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def get_tags(sent):
    return [tag for token, tag in sent]


def get_tokens(sent):
    return [token for token, tag in sent]


def crf_extract_feature_train(_data_train, _data_test):
    _x_train = [get_features(s) for s in _data_train]
    _y_train = [get_tags(s) for s in _data_train]

    _x_test = [get_features(s) for s in _data_test]
    _y_test = [get_tags(s) for s in _data_test]

    return _x_train, _y_train, _x_test, _y_test


if __name__ == '__main__':
    dloader = DataLoader('./data/')
    dloader.load()

    if mini_data:
        dt_train, dt_test = dloader.transform_data(
            sub_train=1000,
            sub_test=100
        )
    else:
        dt_train, dt_test = dloader.transform_data()

    x_train, y_train, x_test, y_test = crf_extract_feature_train(dt_train, dt_test)

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
    )
    crf.fit(x_train, y_train)

    y_predict = crf.predict(x_test)
    same = 0
    _sum = 0
    for st, sp in zip(y_test, y_predict):
        for tt, tp in zip(st, sp):
            if tt == tp:
                same += 1
            _sum += 1

    print("perc: ", same / _sum)
    print('Saving model to >>> crf_pos_model.crf')
    pickle.dump(crf, open('crf_pos_model.crf', mode='wb'))
