"""Unit-tests for the <BioPhiRunner> class"""

from tfold.tools.biophi_runner import BioPhiRunner
import logging


def main():
    # configurations
    oasis_db_path = '/apdcephfs/share_1594716/jonathanwu/Datasets/OAS/OASis/OASis_9mers_v1.db'

    aa_seq_hc = 'EVQLVESGGGLIQPGGSLRLSCAASGLIVSSNYMSWVRQAPGKGLEWVSVLYAGGSTDYAGSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDAAVYGIDVWGQGTTVTVSS'
    aa_seq_lc = (
        'DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYTTPLFTFGPGTKVDIK'
    )

    # test data
    input_data = {
        'hc': {'base': {'seq': aa_seq_hc}, 'feat': {}},
        'lc': {'base': {'seq': aa_seq_lc}, 'feat': {}},
        'cp': {
            'base': {'seq': aa_seq_hc + aa_seq_lc},
            'feat': {},
        },
    }
    # input_data = {'hc': {'base': {'seq': aa_seq_hc}, 'feat': {}}}

    # initialization
    biophi_runner = BioPhiRunner(oasis_db_path)

    # get the sapiens score using BioPhi
    output = biophi_runner.sapiens_score(input_data)

    # get the oasis score using BioPhi
    output = biophi_runner.oasis_score(output)

    # humanized sequence using sapiens
    humanized_sequence = biophi_runner.humanized_sequence(input_data)


if __name__ == '__main__':
    main()
