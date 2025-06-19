"""Unit-tests for the <DockQAssessor> class."""

import os
import logging

from tfold.utils import tfold_init
from tfold.tools import DockQAssessor


def main():
    """Main entry."""

    # configurations
    pdb_fpath_natv1 = '/apdcephfs/share_1594716/fandiwu/Datasets/Antigen-Benchmark/SAbDab-22H1-Cp/pdb.files.native/8dce_H_H_A.pdb'
    pdb_fpath_decy1 = '/apdcephfs/share_1594716/fandiwu/Datasets/Antigen-Benchmark/SAbDab-22H1-Cp/results/pdb.files.decoy.alphafold/8dce_H_H_A.pdb'

    pdb_fpath_natv2 = '/apdcephfs/share_1594716/fandiwu/Datasets/Antigen-Benchmark/SAbDab-22H1-Cp/pdb.files.native/7z1c_F_NA_A.pdb'
    pdb_fpath_decy2 = '/apdcephfs/share_1594716/fandiwu/Datasets/Antigen-Benchmark/SAbDab-22H1-Cp/results/pdb.files.decoy.alphafold/7z1c_F_NA_A.pdb'

    # initialization
    tfold_init(verb_levl='DEBUG')
    assessor = DockQAssessor('/softwares/extras/DockQ-1.0')
    metric_names = assessor.metric_names

    # evaluate the decoy PDB file
    metrics = assessor.run(pdb_fpath_natv1, pdb_fpath_decy1, args='-native_chain1 A -perm1')
    for metric_name in metric_names:
        logging.info('%s: %.4f', metric_name, metrics[metric_name])

    metrics = assessor.run(pdb_fpath_natv2, pdb_fpath_decy2)
    for metric_name in metric_names:
        logging.info('%s: %.4f', metric_name, metrics[metric_name])


if __name__ == "__main__":
    main()
