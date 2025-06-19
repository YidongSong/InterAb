"""CDR region parser."""


from anarci import anarci


class RegionParser():
    """Antibody & TCR region parser."""

    def __init__(self, scheme='imgt'):
        """Constructor function."""

        # setup configurations
        self.scheme = scheme

        # additional configurations
        self.regions = ['FR', 'CDR1', 'CDR2', 'CDR3']
        if self.scheme == 'imgt':
            self.cdr_bnds_dict = {
                'H': {'CDR1': (27, 38), 'CDR2': (56, 65), 'CDR3': (105, 117)},
                'K': {'CDR1': (27, 38), 'CDR2': (56, 65), 'CDR3': (105, 117)},
                'L': {'CDR1': (27, 38), 'CDR2': (56, 65), 'CDR3': (105, 117)},
                'B': {'CDR1': (27, 38), 'CDR2': (56, 65), 'CDR3': (105, 117)},
                'A': {'CDR1': (27, 38), 'CDR2': (56, 65), 'CDR3': (105, 117)},
                'D': {'CDR1': (27, 38), 'CDR2': (56, 65), 'CDR3': (105, 117)},
                'G': {'CDR1': (27, 38), 'CDR2': (56, 65), 'CDR3': (105, 117)},
            }
        elif self.scheme == 'chothia':
            self.cdr_bnds_dict = {
                'H': {'CDR1': (26, 32), 'CDR2': (52, 56), 'CDR3': (95, 102)},
                'L': {'CDR1': (24, 34), 'CDR2': (50, 56), 'CDR3': (89, 97)},
                'K': {'CDR1': (24, 34), 'CDR2': (50, 56), 'CDR3': (89, 97)},
            }
        else:
            raise ValueError(f'unrecognized renumber scheme: {self.scheme}')

    def __get_regions(self, aa_seq_all):
        """Get region annotations (framework or CDRs)."""

        # run ANARCI
        numbering, details, _ = anarci([('null', aa_seq_all)], scheme=self.scheme, output=False)
        aa_seq_sel = ''.join([x[1] for x in numbering[0][0][0] if x[1] != '-'])
        chn_type = details[0][0]['chain_type']
        cdr_bnds = self.cdr_bnds_dict[chn_type]
        assert aa_seq_sel in aa_seq_all
        n_resds_lp = aa_seq_all.index(aa_seq_sel)  # padding on the left side
        n_resds_rp = len(aa_seq_all) - len(aa_seq_sel) - n_resds_lp  # padding on the right side

        # parse ANARCI's outputs
        resd_list = []
        if n_resds_lp != 0:
            resd_list.extend([(x, None) for x in aa_seq_all[:n_resds_lp]])
        for (idx_resd, _), resd_name in numbering[0][0][0]:
            if resd_name == '-':
                continue
            rgn_name = 'FR'  # default region name
            for cdr_name, (idx_resd_beg, idx_resd_end) in cdr_bnds.items():
                if idx_resd_beg <= idx_resd <= idx_resd_end:
                    rgn_name = cdr_name
                    break
            resd_list.append((resd_name, rgn_name))
        if n_resds_rp != 0:
            resd_list.extend([(x, None) for x in aa_seq_all[-n_resds_rp:]])
        assert ''.join([x[0] for x in resd_list]) == aa_seq_all

        return resd_list

    def get_regions(self, aa_seq):
        """Get region annotations (framework or CDRs)."""

        resd_list = self.__get_regions(aa_seq)
        regions = {'fr': [], 'cdr1': [], 'cdr2': [], 'cdr3': [], 'cdr': []}

        for idx, (_, region) in enumerate(resd_list):
            if region is None:
                continue
            regions[region.lower()].append(idx)
            if region.lower().startswith('cdr'):
                regions['cdr'].append(idx)

        return regions
