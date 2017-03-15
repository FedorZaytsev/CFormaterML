# CFormaterML

Formater for C language using machine learning idea.

## Introduction
The aim of this project is to demonstrate an idea which use machine
learning algorithms to format code. This is academic project only,
there are no guarantees that code will be still compilable after
formatting.

## Installation
1. Install cparser (https://github.com/FedorZaytsev/cparser)
2. Install sklearn, numpy, pandas (pip3 install sklearn numpy pandas)
3. Clone project (git clone https://github.com/FedorZaytsev/CFormaterML)

## Usage
```buildoutcfg
usage: main.py [-h] [--train TRAIN]
               [--classifier_filename CLASSIFIER_FILENAME]
               [--clftab {solving_tree,kneighbors,svm,random_forest}]
               [--clfspace {solving_tree,kneighbors,svm,random_forest}]
               [--clfnl {solving_tree,kneighbors,svm,random_forest}]
               [--load_clfs CLFS]
               [file]
               
Formatter for C language

positional arguments:
  file                  File to process

optional arguments:
  -h, --help            show this help message and exit
  --train TRAIN         Train classifier on that folder
  --classifier_filename CLASSIFIER_FILENAME
                        In which file save classifier
  --clftab {solving_tree,kneighbors,svm,random_forest}
                        Type of classifier for tabs
  --clfspace {solving_tree,kneighbors,svm,random_forest}
                        Type of classifier for spaces
  --clfnl {solving_tree,kneighbors,svm,random_forest}
                        Type of classifier for newlines
  --load_clfs CLFS      Load previously saved classifiers
```

Generally, to use that program, you have to do 2 steps:
1. Train classificators:

    ```python3 ./main.py --train FOLDER_TO_TRAIN_ON```
    
    Additionally, you can specify in which file classificators
    will be saved. Formally, this step can be skipped, however,
    training classificators is a prolonged task, so it is recomended
    to save trained classificators
    
2. Format source code:
    
    ```python3 ./main.py --load_clfs STORED_CLFS SOME_FILE```
    
    This format file 'SOME_FILE' at print it to output. STORED_CLFS
    is a path to stored classificators.
    
    
## I don't want to read all those, just how to start that program!

```buildoutcfg
mkdir cformater
cd cformater
git clone https://github.com/rusphantom/cparser.git
cd cparser/
pip3 install .
cd ..
git clone https://github.com/FedorZaytsev/CFormaterML.git
cd CFormaterML/
pip3 install sklearn numpy pandas
git clone https://github.com/torvalds/linux.git
python3 ./main.py --train ./linux/kernel/
python3 ./main.py --load_clfs classificators.data ./adlist_test.c
```

This will install all required libs, and format test file adlist_test.c


## What about accuracy?
|               | Newline | Space   | Tabulation |
| ------------- | ------- | ------- | ---------- |
| Solving tree  | 0.97415 | 0.98445 | 0.95861    |
| K-neighbors   | 0.96799 | 0.98526 | 0.95469    |
| SVM           | 0.97029 | 0.98478 | 0.95368    |
| Random Forest | 0.97639 | 0.99017 | 0.96819    |

## Example?

Source code:
```buildoutcfg
static int audit_log_config_change(char *function_name, u32 new, u32 old,
				   int allow_changes)
{
struct audit_buffer *ab;
	int rc = 0;

	        ab = audit_log_start(NULL, GFP_KERNEL, AUDIT_CONFIG_CHANGE);
if (unlikely(!ab))
return rc;
	        audit_log_format(ab, "%s=%u old=%u", function_name, new, old);
	audit_log_session_info(ab);
	        rc

	        =
	        audit_log_task_context(ab);
	if (rc

	)
allow_changes = 0; /* Something weird, deny request */


audit_log_format
(
ab, " res=%d", allow_changes);
	audit_log_end(ab);
	return rc;
}

```

Formatted code:
```buildoutcfg
static int audit_log_config_change(char *function_name, u32 new, u32 old, 			int allow_changes)
{
	struct audit_buffer *ab;
	int rc = 0;

	ab = audit_log_start(NULL, GFP_KERNEL, AUDIT_CONFIG_CHANGE);
	if (unlikely(!ab))
		return rc;

	audit_log_format(ab, "%s=%u old=%u",
 function_name, new, old);
	audit_log_session_info(ab);
	rc = audit_log_task_context(ab);
	if (rc)
		allow_changes = 0;

 	/* Something weird, deny request */
	audit_log_format(ab, " res=%d", allow_changes);
	audit_log_end(ab);
	return rc;
}
```

## Pluses
* None

## Minuses
* Cparser is not working well - support only pure C,
some parts of code can't be parsed with it.
* Classificators are not well tunned
* There are no guarantees that formatting will not change
code behaviour - sometimes space classificator can make a mistake
and drop space between two identificators (can be fixed)

## Trained classificators
Classificators trained on Linux kernel: https://www.dropbox.com/s/ucyo19143xqy8g9/linux_classificators.data?dl=0

## Settings
config.py contains some interesting settings:
* tags_newline, tags_space, tags_tab - contain a list of features,
on which classificators train.
* parent_count - count of parent nodes which will be used as features.
* categorial_features - list of categorial features.
* balance - is we balance train/test sets.
* print_prediction_for_* - print predictions for each object or not. Sometimes it is usefull.
* print_ast - print AST of files after parsing.
* debug_mode - debug mode. Now unused I believe.
* files2process - how many files to process. If not set use
up to 1000000 files to train on. If set then use no more than that number.


