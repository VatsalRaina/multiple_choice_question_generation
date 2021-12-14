import argparse
import os
import sys

from numpy.lib.arraysetops import unique

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--questions_path', type=str,  help='Specify path to generated questions')
parser.add_argument('--contexts_path', type=str,  help='Specify path to contexts')


def organise_data(questions, contexts):
    organised_data = []
    for question, context in zip(questions, contexts):
        first_sep_pos = question.find("[SEP]")
        qu = question[:first_sep_pos]
        opts = []
        validSEP = True
        sep_pos = first_sep_pos
        while validSEP:
            question = question[sep_pos+6:]
            sep_pos = question.find("[SEP]")
            if sep_pos == -1:
                validSEP = False
            else:
                opt = question[:sep_pos]
                opts.append(opt)
        curr_point = {'question': qu, 'context': context, 'options':opts, 'label':0}
        organised_data.append(curr_point)
    return organised_data

def got_four_opts(test_data):
    num_valid = 0
    for ex in test_data:
        unique_opts = []
        for opt in ex['options']:
            if opt not in unique_opts:
                unique_opts.append(opt)
        if len(unique_opts) == 4:
            num_valid += 1
    return num_valid / len(test_data)


def get_unanswerability():
    pass

def get_accuracy():
    pass

def get_complexity():
    pass

def main(args):
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    with open(args.questions_path, 'r') as f:
        all_gen_questions = [a.rstrip() for a in f.readlines()]

    with open(args.contexts_path, 'r') as f:
        all_contexts = [a.rstrip() for a in f.readlines()]

    organised_data = organise_data(all_gen_questions, all_contexts)

    frac_four_opts = got_four_opts(organised_data)
    print("Fraction of samples with 4 unique options:", frac_four_opts)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)