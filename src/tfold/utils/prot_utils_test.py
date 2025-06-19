import os
from timeit import default_timer as timer
from tfold.utils import parse_fas_file_mult

data_dir = '/data/jonathanwu/for_royrong'

path = os.path.join(data_dir, 'uniref50_1m.fasta')
print(f'input file: {path}')
time_beg = timer()
aa_seq_dict = parse_fas_file_mult(path, is_ordered=True)
print(f'parsing time: {timer() - time_beg:.2f} (s) - ordered')
time_beg = timer()
aa_seq_dict = parse_fas_file_mult(path, is_ordered=False)
print(f'parsing time: {timer() - time_beg:.2f} (s) - not-ordered')

path = os.path.join(data_dir, 'uniref50_1m.fasta.gz')
print(f'input file: {path}')
time_beg = timer()
aa_seq_dict = parse_fas_file_mult(path, is_ordered=True)
print(f'parsing time: {timer() - time_beg:.2f} (s) - ordered')
time_beg = timer()
aa_seq_dict = parse_fas_file_mult(path, is_ordered=False)
print(f'parsing time: {timer() - time_beg:.2f} (s) - not-ordered')

path = os.path.join(data_dir, 'uniref50_1m_split.fasta')
print(f'input file: {path}')
time_beg = timer()
aa_seq_dict = parse_fas_file_mult(path, is_ordered=True)
print(f'parsing time: {timer() - time_beg:.2f} (s) - ordered')
time_beg = timer()
aa_seq_dict = parse_fas_file_mult(path, is_ordered=False)
print(f'parsing time: {timer() - time_beg:.2f} (s) - not-ordered')

path = os.path.join(data_dir, 'uniref50_1m_split.fasta.gz')
print(f'input file: {path}')
time_beg = timer()
aa_seq_dict = parse_fas_file_mult(path, is_ordered=True)
print(f'parsing time: {timer() - time_beg:.2f} (s) - ordered')
time_beg = timer()
aa_seq_dict = parse_fas_file_mult(path, is_ordered=False)
print(f'parsing time: {timer() - time_beg:.2f} (s) - not-ordered')
