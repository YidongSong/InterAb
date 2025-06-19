"""Unit-tests for the <A3mParser> class."""

import os
import logging

from tfold.utils import tfold_init
from tfold.tools import A3mParser


def main():
    """Main entry."""

    # configurations
    prot_id = 'T1024-D1'
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(curr_dir, 'examples')
    a3m_fpath = os.path.join(data_dir, f'{prot_id}.a3m')

    # initialization
    tfold_init(verb_levl='DEBUG')

    # compare parsing results of A3M files
    parser_v1 = A3mParser(version='v1')
    parser_v2 = A3mParser(version='v2')
    records_v1 = parser_v1.run(a3m_fpath)
    records_v2 = parser_v2.run(a3m_fpath)
    for record_v1, record_v2 in zip(records_v1, records_v2):
        assert (record_v1[0] == record_v2[0]) and (record_v1[1] == record_v2[1]), \
            f'inconsistent parsing results: {record_v1} vs. {record_v2}'
    logging.info('# of records in the A3M file: %d', len(records_v1))


if __name__ == '__main__':
    main()
