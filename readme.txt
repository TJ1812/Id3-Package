This package is developed using python 3

You need to install "pandas" for running the decision_tree.py successfully.

cmd:-
pip install pandas

open console in the directory of installation.

to run the file decision_tree.py, write command of type :-

python decision_tree.py "L" "K" "path_to_training_set (csv)" "path_to_validation_set (csv)" "path_to_test_set (csv)" "print_tree(yes/no)"


eg.

python decision_tree.py 10 10 "data_sets2/training_set.csv" "data_sets2/validation_set.csv" "data_sets2/test_set.csv" "no"
python decision_tree.py 100 10 "data_sets1/training_set.csv" "data_sets1/validation_set.csv" "data_sets1/test_set.csv" "yes"