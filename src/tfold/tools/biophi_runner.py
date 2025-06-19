import subprocess
import tempfile
import pandas as pd
from tfold.utils import parse_fas_file_mult


class BioPhiRunner:  # pylint: disable=too-few-public-methods
    """
    BioPhi Runner for antibody humanization using Sapiens and humanness evaluation using OASis.

    * Note: Users should install BioPhi and download OASIS_DB first.
    """

    def __init__(self, oasis_db_path, scheme='imgt', cdr_definition='imgt', verbose=True):
        """Constructor function."""

        self.oasis_db_path = oasis_db_path
        self.scheme = scheme
        self.cdr_definition = cdr_definition
        self.verbose = verbose

        self.chn_types = {'hc': 'VH', 'lc': 'VL'}
        self.tmp_path = tempfile.mkdtemp(prefix='tmpResult4BioPhi')

    def parepare_sequence(self, inputs):
        """Parepare the input for BioPhiRunner.

        Args:
        * inputs: dict of input tensor (amino-acid sequence)

        Returns:
        * sequence_file: path of input file for BioPhiRunner
        """
        chn_types = sorted([x for x in inputs if x in self.chn_types])
        with open(f'{self.tmp_path}/seq.fasta', 'w') as F:
            for chn_type in chn_types:
                F.write(f'>seq_{self.chn_types[chn_type]}\n')
                F.write(f'{inputs[chn_type]["base"]["seq"]}\n')

    def humanized_sequence(self, inputs):
        """humanized sequence using sapiens

        Args:
        * inputs: dict of input tensor (amino-acid sequence)

        Returns:
        * outputs: dict of output tensor (amino-acid sequence)
        """
        self.parepare_sequence(inputs)
        cmd_str = f'biophi sapiens {self.tmp_path}/seq.fasta --fasta-only --output {self.tmp_path}/humanized.fasta'
        subprocess.check_output(cmd_str, shell=True)
        hum_seq = parse_fas_file_mult(f'{self.tmp_path}/humanized.fasta')
        output = {}
        for chn_type in inputs:
            output[chn_type] = {}

        for key in hum_seq:
            if key.split()[0].endswith('VH'):
                output['hc'] = {'base': {'seq': hum_seq[key]}, 'feat': {}}
            if key.split()[0].endswith('VL'):
                output['lc'] = {'base': {'seq': hum_seq[key]}, 'feat': {}}
        return output

    def sapiens_score(self, inputs):
        """Get the sapiens score using BioPhi
        Args:
        * inputs: dict of input tensor (amino-acid sequence)

        Returns:
        * outputs: dict of output tensor (amino-acid sequence, prob_matrix)
        """
        self.parepare_sequence(inputs)
        cmd_str = (
            f'biophi sapiens {self.tmp_path}/seq.fasta --scheme {self.scheme} --cdr-definition {self.cdr_definition} --mean-score-only --output {self.tmp_path}/sapien_scores.csv'
        )
        subprocess.check_output(cmd_str, shell=True)
        chn_types = sorted([x for x in inputs if x in self.chn_types])

        for num, chn_type in enumerate(chn_types):
            inputs[chn_type]['feat']['sapiens_score'] = pd.read_csv(f'{self.tmp_path}/sapien_scores.csv').iloc[num][
                'sapiens_score'
            ]
        return inputs

    # get the OASis score
    def oasis_score(self, inputs, min_precent_subjects=10):
        """Get the OASis score using BioPhi (OASis percentile & OASis identity)
        Args:
        * inputs: dict of input tensor (amino-acid sequence)
        * min-precent-subjects: Minimum percent of OAS subjects to consider peptide human (1, 10, 50, 90, default=10)

        Returns:
        * outputs: dict of output tensor (amino-acid sequence, oasis_percentile, oasis_id)
        *
        """
        self.parepare_sequence(inputs)
        cmd_str = f'biophi oasis {self.tmp_path}/seq.fasta --scheme {self.scheme} --cdr-definition {self.cdr_definition} --min-percent-subjects {min_precent_subjects} --oasis-db {self.oasis_db_path} --output {self.tmp_path}/oasis_score.xlsx'

        if self.verbose:
            subprocess.check_output(cmd_str, shell=True)
        else:
            subprocess.run(cmd_str, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True, check=True)

        chn_types = sorted([x for x in inputs if x in self.chn_types])
        oasis_result = pd.read_excel(f'{self.tmp_path}/oasis_score.xlsx')
        for chn_type in chn_types:
            if chn_type == 'hc':
                inputs[chn_type]['feat']['oasis_perc'] = oasis_result.iloc[0]['Heavy OASis Percentile']
                inputs[chn_type]['feat']['oasis_id'] = oasis_result.iloc[0]['Heavy OASis Identity']
            elif chn_type == 'lc':
                inputs[chn_type]['feat']['oasis_perc'] = oasis_result.iloc[0]['Light OASis Percentile']
                inputs[chn_type]['feat']['oasis_id'] = oasis_result.iloc[0]['Light OASis Identity']
        if len(chn_types) > 1:
            inputs['cp']['feat']['oasis_perc'] = oasis_result.iloc[0]['OASis Percentile']
            inputs['cp']['feat']['oasis_id'] = oasis_result.iloc[0]['OASis Identity']
        return inputs
