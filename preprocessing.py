from cparser import Node, lexer, analyzer
import pandas
from config import *
from sklearn import feature_extraction
import os


def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r


#count element in specified direction and skip some other
def count(l, node, type, skip=None, direction="forward"):
    if skip is None:
        skip = []
    dir = 1
    if direction == "forward":
        dir = 1
    elif direction == "backward":
        dir = -1
    else:
        raise Exception("wrong direction {}".format(direction))

    count = 0
    lexem = l.getByIdx(node.idx, dir, False)
    while lexem.type in [type] + skip:
        if lexem.type == type:
            count += 1
        lexem = l.getByIdx(lexem.idx, dir, False)
    return count


def countElementsUntil(l, node, until, skip, direction="forward"):
    dir = 1
    if direction == "forward":
        dir = 1
    elif direction == "backward":
        dir = -1
    else:
        raise Exception("wrong direction {}".format(direction))

    if type(until) != "list":
        until = [until]

    count = 0
    lexem = l.getByIdx(node.idx, dir, False)
    while lexem.type != "EOF" and lexem.type not in until:
        if lexem.type not in skip:
            count += 1
        lexem = l.getByIdx(lexem.idx, dir, False)
        if count> 100:
            break

    return count


def getLexems(l, a, ast, X=None, y=None):
    if X is None:
        X = []
    if y is None:
        y = []

    def getTokenByOffset(idx, offset):
        nextlexem = l.getByIdx(idx, offset, False)
        while nextlexem.type in ['NEWLINE', 'SPACE', 'TAB']:
            idx += offset
            nextlexem = l.getByIdx(idx, offset, False)
        return nextlexem

    for node in ast:
        if type(node) is lexer.Lexem:
            nextlexem = getTokenByOffset(node.idx, 1)
            prevlexem = getTokenByOffset(node.idx, -1)

            parent = node.parent
            parent_count = 0
            while parent is not None:
                parent_count += 1
                parent = parent.parent

            element = {
                'rep': node.representation,
                'tag': node.type,
                'tag_next': nextlexem.type,
                'tag_prev': prevlexem.type,
                'size_tag': len(node.parent.children),
                'size_tag_next': 0 if nextlexem.type == 'EOF' or nextlexem.parent is None else len(nextlexem.parent.children),
                'len_until_lexem': node.position.column,
                'count_lexems': countElementsUntil(l, node, "NEWLINE", ["SPACE", "TAB"], "backward"),
                'parent_count': parent_count
            }



            parent = node.parent
            for i in range(gconfig['parent_count']):
                if parent is None:
                    element['parent_{}'.format(i+1)] = 'UNKNOWN'
                else:
                    element['parent_{}'.format(i+1)] = parent.name
                    parent = parent.parent
            X.append(element)

            newlines = count(l, node, "NEWLINE", ["SPACE", "TAB"], "forward")
            spaces = count(l, nextlexem, "SPACE", ["TAB"], "backward")
            tabs = count(l, nextlexem, "TAB", ["SPACE"], "backward")

            y.append({'newlines':newlines, 'spaces':spaces, 'tabs':tabs})

            if newlines > 10 and spaces > 10 and tabs > 10:
                raise Exception("Too many tabs, spaces or newlines. Possible error in calculations?{} {}".format(node.position, node))


        elif type(node) is Node.Node:

            #skip error nodes
            if node.name != 'ERROR':
                X, y = getLexems(l, a, node, X, y)

    return X, y


def printParsingInfo(a, file, allskiped, allparsed):
    if len(a.errors) > 0:
        size = 0
        for e in a.errors:
            size += len(e.children[0].representation)
        allskiped += size
        allparsed += os.stat(file).st_size
        print("parsed {:.2f}%".format(100 - size / os.stat(file).st_size * 100))
    else:
        print("parsed 100%")

    return allskiped, allparsed

def getData(foldername):
    files = list_files(foldername)
    df = None
    allskiped = 0
    allparsed = 0
    countparsed = 0
    max_count = gconfig['files2process'] if 'files2process' in gconfig else 1000000
    if gconfig['debug_mode']:
        max_count = 2
    for file in files:
        if file[-1] != 'c':
            continue
        if max_count <= 0:
            break
        print("opening file {}".format(file))
        try:
            data = open(file).read()
            l = lexer.Lexer(data)
            a = analyzer.Analyzer(l)
            ast = analyzer.normalizeAST(a.parse())

            allskiped, allparsed = printParsingInfo(a, file, allskiped, allparsed)


            Xfile, yfile = getLexems(l, a, ast)
            assert len(Xfile) == len(yfile)

            partdf = pandas.DataFrame([list(Xfile[i].values()) + list(yfile[i].values()) for i in range(len(Xfile))],
                                      columns=list(Xfile[0].keys())+list(yfile[0].keys()))

            if df is None:
                df = partdf
            else:
                df = df.append(partdf, ignore_index=True)
            countparsed += 1

        except UnicodeDecodeError:
            pass

        max_count -= 1

    if allparsed == 0:
        raise Exception("No appropriate files found")

    print("parsed {0} files\n{1:.2f}% skiped".format(countparsed, allskiped / allparsed * 100))

    return df


def onehot(csv):
    records = csv[gconfig['categorial_features']].to_dict(orient='records')
    dv = feature_extraction.DictVectorizer(separator='_', sparse=False)
    dv.fit(records)
    return dv

def onehottranform(csv, encoder):
    columns = set(gconfig['categorial_features']).intersection(csv.columns)
    #print("onehottranform", columns)
    data = encoder.transform(csv[list(columns)].to_dict(orient='records'))

    csv1 = csv.drop(columns, axis=1)
    return pandas.concat([csv1, pandas.DataFrame(data, columns=encoder.feature_names_, index=csv1.index)], axis=1)




