import numpy as np
import random
import  pickle

num_train = 100000
num_val = 10000
num_test = 20000

step_num = 24
elem_num = 26 + 10 + 1

x_train = np.zeros([num_train, step_num * 2 + 3], dtype=np.float32)
x_val = np.zeros([num_val, step_num * 2 + 3], dtype=np.float32)
x_test = np.zeros([num_test, step_num * 2 + 3], dtype=np.float32)

y_train = np.zeros([num_train], dtype=np.float32)
y_val = np.zeros([num_val], dtype=np.float32)
y_test = np.zeros([num_test], dtype=np.float32)


def get_one_hot(c):
    a = np.zeros([elem_num])
    if ord('a') <= ord(c) <= ord('z'):
        a[ord(c) - ord('a')] = 1
    elif ord('0') <= ord(c) <= ord('9'):
        a[ord(c) - ord('0') + 26] = 1
    else:
        a[-1] = 1
    return a

def get_one_hot2(c):
    if ord('a') <= ord(c) <= ord('z'):
        return ord(c) - ord('a')+10
    elif ord('0') <= ord(c) <= ord('9'):
        return ord(c) - ord('0')

    return elem_num -1


def generate_one():
    a = np.zeros([step_num * 2 + 3, elem_num])
    d = {}
    st = ''

    for i in range(0, step_num):
        c = random.randint(0, 25)
        while c in d:
            c = random.randint(0, 25)
        b = random.randint(0, 9)
        d[c] = b
        s, t = chr(c + ord('a')), chr(b + ord('0'))
        st += s + t
        a[i*2] = get_one_hot(s)
        a[i*2+1] = get_one_hot(t)

    s = random.choice(list(d.keys()))
    t = chr(s + ord('a'))
    r = chr(d[s] + ord('0'))
    a[step_num * 2] = get_one_hot('?')
    a[step_num * 2 + 1] = get_one_hot('?')
    a[step_num * 2 + 2] = get_one_hot(t)
    st += '??' + t + r
    e = get_one_hot(r)
    return a, e

def generate_one2():
    a = np.zeros([step_num * 2 + 3])
    d = {}
    st = ''

    for i in range(0, step_num):
        c = random.randint(0, 25)
        while c in d:
            c = random.randint(0, 25)
        b = random.randint(0, 9)
        d[c] = b
        s, t = chr(c + ord('a')), chr(b + ord('0'))
        st += s + t
        a[i*2] = get_one_hot2(s)
        a[i*2+1] = get_one_hot2(t)

    s = random.choice(list(d.keys()))
    t = chr(s + ord('a'))
    r = chr(d[s] + ord('0'))
    a[step_num * 2] = get_one_hot2('?')
    a[step_num * 2 + 1] = get_one_hot2('?')
    a[step_num * 2 + 2] = get_one_hot2(t)
    st += '??' + t + r
    e = get_one_hot2(r)
    return a, e


if __name__ == '__main__':
    for i in range(0, num_train):
        x_train[i], y_train[i] = generate_one2()

    for i in range(0, num_test):
        x_test[i], y_test[i] = generate_one2()

    for i in range(0, num_val):
        x_val[i], y_val[i] = generate_one2()

    d = {
        'x_train': x_train,
        'x_test': x_test,
        'x_val': x_val,
        'y_train': y_train,
        'y_test': y_test,
        'y_val': y_val
    }
    with open('./associative-retrieval.pkl', 'wb') as f:
        pickle.dump(d, f, protocol=2)