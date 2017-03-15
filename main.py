import argparse
import dumper
import learning


def main():
    parser = argparse.ArgumentParser(description='Formatter for C language')
    parser.add_argument('--train', dest='train', type=str, help='Train classifier on that folder')
    parser.add_argument('--classifier_filename', dest='classifier_filename', type=str, default='linux_classificators.data',
                        help='In which file save classifier')
    parser.add_argument('--clftab', dest='clftab', type=str,
                        choices=['solving_tree', 'kneighbors', 'svm', 'random_forest'], default='random_forest',
                        help='Type of classifier for tabs')
    parser.add_argument('--clfspace', dest='clfspace', type=str,
                        choices=['solving_tree', 'kneighbors', 'svm', 'random_forest'], default='random_forest',
                        help='Type of classifier for spaces')
    parser.add_argument('--clfnl', dest='clfnewline', type=str,
                        choices=['solving_tree', 'kneighbors', 'svm', 'random_forest'], default='random_forest',
                        help='Type of classifier for newlines')
    parser.add_argument('--load_clfs', dest='clfs', type=str, help='Load previously saved classifiers')
    parser.add_argument('file', help='File to process', nargs='?')
    args = parser.parse_args()

    clfs = None
    if args.train is not None:
        clfs = learning.generate_classifiers(args)
        dumper.dump(clfs, learning.vectorizer, args.classifier_filename)

    elif args.clfs is not None:
        data = dumper.load(args.clfs)
        clfs = data['classifiers']
        learning.vectorizer = data['vectorizer']

    if args.file is not None:
        print("Processing file...")
        data = learning.format_file(args.file, clfs['newline'], clfs['space'], clfs['tab'])
        print(data)






if __name__ == '__main__':
    main()