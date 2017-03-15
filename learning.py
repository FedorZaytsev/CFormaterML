from preprocessing import *
from cparser import debug
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors, svm, metrics
from sklearn.model_selection import train_test_split
import numpy as np

vectorizer = None

def get_keys(df, type):
    data = df[[el for el in gconfig['tags_{}'.format(type)]]]

    df = onehottranform(data, vectorizer)
    return df


def solvingTree(keys, values, type):
    classifier = DecisionTreeClassifier(random_state=241)
    classifier.fit(get_keys(keys, type), values)

    return classifier


def kneighbors(keys, values, type):
    k = 5
    classifier = neighbors.KNeighborsClassifier(k)
    classifier.fit(get_keys(keys, type), values)

    return classifier


def svc(keys, values, type):
    classifier = svm.SVC(kernel='linear', random_state=241, C=1.0, verbose=True)
    classifier.fit(get_keys(keys, type), values)

    print(classifier.coef_)

    return classifier


def randomForest(keys, values, type):
    classifier = RandomForestClassifier(n_estimators=100, random_state=241, verbose=False)
    classifier.fit(get_keys(keys, type), values)
    return classifier


def get_classificator_by_name(args, name):
    name = getattr(args, 'clf{}'.format(name))
    if name == 'solving_tree':
        return solvingTree
    elif name == 'kneighbors':
        return kneighbors
    elif name == 'svm':
        return svc
    elif name == 'random_forest':
        return randomForest
    else:
        raise Exception('Wrong classificator name')


def balanced_subsample(x,y,subsample_size=1.0):
    np.random.seed(241)

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        if len(elems) < 5:
            continue
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            this_xs = this_xs.reindex(np.random.permutation(this_xs.index))

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = pandas.concat(xs)
    ys = pandas.Series(data=np.concatenate(ys), name='target')

    return xs,ys


def test_accuracy(clfs, df):
    mtrcs = {}
    for name in ['newline', 'space', 'tab']:
        X = None
        y = None
        if gconfig['balance']['test']:
            X, y = balanced_subsample(df.drop(['newlines', 'spaces', 'tabs'], axis=1), df[name+'s'])
            print("subsample test size for {} is {}".format(name, X.shape))
        else:
            X = df.drop(['newlines', 'spaces', 'tabs'], axis=1)
            y = df[name + 's']
        pred = clfs[name].predict(get_keys(X, name))
        mtrcs[name] = metrics.f1_score(y, pred, average='weighted')

    print("Metrics {}".format(mtrcs))


def generate_classifiers(args):
    df = getData(args.train)

    global vectorizer
    vectorizer = onehot(df)

    print("dataset size {}".format(df.shape))
    print("columns", df.columns)

    df_train, df_test = train_test_split(df, test_size=0.3, random_state=241)

    clfs = {}
    for name in ['tab', 'space', 'newline']:
        X = None
        y = None
        if gconfig['balance']['train']:
            X, y = balanced_subsample(df_train.drop(['newlines', 'spaces', 'tabs'], axis=1), df_train[name + 's'])
            print("subsample train size for {} is {}".format(name, X.shape))
        else:
            X = df_train.drop(['newlines', 'spaces', 'tabs'], axis=1)
            y = df_train[name + 's']

        clfs[name] = get_classificator_by_name(args, name)(X, y, name)

    print("classifiers ready")

    test_accuracy(clfs, df_test)
    return clfs


def predict(classifier, el, type, symbol):
    el = pandas.DataFrame([el.to_dict().values()], columns=el.to_dict().keys())
    key = get_keys(el, type)

    result = classifier.predict(key)
    if gconfig['print_prediction_for_each'] or gconfig['print_prediction_for_each_{}'.format(type)]:
        print("{} = {}".format(el.to_dict(), result))

    return symbol * result[0] if result[0] > 0 else ''


def calculateLenUntilLexem(results):
    length = 0
    for i in range(len(results)-1, -1, -1):
        if len(results[i])>0 and results[i][0] == '\n':
            return length
        length += len(results[i])
    return length


def process_format_file(filename, clfnl, clfsp, clftb):
    l = lexer.Lexer(open(filename).read())
    a = analyzer.Analyzer(l)
    ast = analyzer.normalizeAST(a.parse())


    if gconfig['print_ast']:
        debug.printNode(ast)
    X, _ = getLexems(l, a, ast)
    X = pandas.DataFrame([el.values() for el in X], columns=X[0].keys())

    ynewline = []
    yspace = []
    ytab = []


    predictedFile = []
    countLexems = 0
    for idx, row in X.iterrows():
        row['len_until_lexem'] = calculateLenUntilLexem(predictedFile)
        row['count_lexems'] = countLexems
        predictedFile.append(row['rep'])
        part_nl = predict(clfnl, row, 'newline', '\n')
        part_sp = predict(clfsp, row, 'space', ' ')
        part_tb = predict(clftb, row, 'tab', '\t')
        ynewline.append(len(part_nl))
        yspace.append(len(part_sp))
        ytab.append(len(part_tb))
        predictedFile.append(part_nl)
        predictedFile.append(part_sp)
        predictedFile.append(part_tb)
        if len(part_nl) > 0:
            countLexems = 0
        else:
            countLexems += 1

    return predictedFile, ynewline, yspace, ytab


def format_file(filename, clfnl, clfsp, clftb):
    pred, _, _, _ = process_format_file(filename, clfnl, clfsp, clftb)
    return ''.join(pred)


def compare(filename1, filename2, clfnl, clfsp, clftb):
    _, ynl, ysp, ytb = process_format_file(filename1, clfnl, clfsp, clftb)

    l = lexer.Lexer(open(filename2).read())
    a = analyzer.Analyzer(l)
    ast = analyzer.normalizeAST(a.parse())

    _, yfile = getLexems(l, a, ast)

    yfile = pandas.DataFrame([list(yfile[i].values()) for i in range(len(yfile))], columns=list(yfile[0].keys()))

    metricNL = metrics.f1_score(yfile['newlines'], ynl, average='weighted')
    metricSP = metrics.f1_score(yfile['spaces'], ysp, average='weighted')
    metricTB = metrics.f1_score(yfile['tabs'], ytb, average='weighted')

    return {"newline": metricNL, "space": metricSP, "tab": metricTB}

