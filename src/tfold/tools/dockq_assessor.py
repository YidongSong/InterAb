"""Assessor for decoy PDB files for multimer."""

import os
import subprocess
import logging


class DockQAssessor():
    """Assessor docking quality for protein-protein docking models using DockQ.
    Available evaluation metrics:
    > DockQ
    > Fnat
    > LRMS
    > iRMS
    """
    def __init__(self, dockq_path):
        """Constructor function."""

        # setup configurations
        if not os.path.exists(os.path.join(dockq_path, 'DockQ.py')):
            raise ValueError(f'dockq_path: {dockq_path} is not exists')

        self.dockq_path = dockq_path
        self.metric_names = ['DockQ', 'Fnat', 'iRMS', 'LRMS']


    def run(self, pdb_fpath_natv, pdb_fpath_decy, args=''):
        """Run the assessor to compute DockQ score.

        Args:
        * pdb_fpath_natv: path to the native PDB file
        * pdb_fpath_decy: path to the decoy PDB file

        Returns:
        * metrics_dict: dict of evaluation metric values (key: metric name)
        """

        py_fpath = os.path.join(self.dockq_path, 'DockQ.py')

        cmd_str = ' '.join([py_fpath, pdb_fpath_decy, pdb_fpath_natv, args])
        try:
            cmd_out = subprocess.check_output(cmd_str, shell=True)
            line_strs = cmd_out.decode('utf-8').split('\n')
        except Exception as e:
            logging.error(cmd_str)
            raise e

        metrics = {}
        for line_str in line_strs:
            sub_strs = line_str.split()
            if (len(sub_strs) >= 2) and (sub_strs[0] in self.metric_names):
                metrics[sub_strs[0]] = float(sub_strs[1])

        return metrics
