import pickle


def dump(clfs, vectorizer, name):
    with open(name, 'wb') as f:
        f.write('fedorzaytsev'.encode())
        f.write(pickle.dumps({
            'classifiers': clfs,
            'vectorizer': vectorizer
        }))

    print("Saved into {}!".format(name))


def load(name):
    with open(name, 'rb') as f:
        expected_header = 'fedorzaytsev'.encode()
        header = f.read(len(expected_header))
        if expected_header != header:
            raise Exception("Wrong file")

        obj = pickle.loads(f.read())
        print("Loaded!")
        return obj









