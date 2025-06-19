"""Unit-tests for the <RegionParser>."""


from tfold.utils import tfold_init
from tfold.tools import RegionParser


def main():
    """Main entry."""

    # configurations

    aa_seqs = [
        'EVQLQQSGAEVVRSGASVKLSCTASGFNIKDYYIHWVKQRPEKGLEWIGWIDPEIGDTEYVPKFQGKATMTADTSSNTAYLQLSSLTSEDTAVYYCNAGHDYDRGRFPYWGQGTLVTVSA',
        'DIVMTQSQKFMSTSVGDRVSITCKASQNVGTAVAWYQQKPGQSPKLMIYSASNRYTGVPDRFTGSGSGTDFTLTISNMQSEDLADYFCQQYSSYPLTFGAGTKLELK',
        'QVRQSPQSLTVWEGETTILNCSYEDSTFDYFPWYRQFPGKSPALLIAISLVSNKKEDGRFTIFFNKREKKLSLHITDSQPGDSATYFCAATGSFNKLTFGAGTRLAVSP',
        'AVTQSPRNKVAVTGGKVTLSCNQTNNHNNMYWYRQDTGHGLRLIHYSYGAGSTEKGDIPDGYKASRPSQENFSLILELATPSQTSVYFCASGGQGRAEQFFGPGTRLTVL',
    ]

    region_parser = RegionParser()

    # initialization
    tfold_init()

    for aa_seq in aa_seqs:
        regions = region_parser.get_regions(aa_seq)

        print(aa_seq, regions)


if __name__ == '__main__':
    main()
