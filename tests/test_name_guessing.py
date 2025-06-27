from xifdr.utils.guess_columns import guess_column_names


def test_xisearch2():
    columns = [
        'match_score', 'top_ranking', 'fdr_group',
        'sequence_p1', 'sequence_p2',
        'link_pos_p1', 'link_pos_p2',
        'protein_p1', 'protein_p2',
        'protein_link_p1', 'protein_link_p2',
        'decoy_p1', 'decoy_p2',
        'base_sequence_p1', 'base_sequence_p2',
        'mass_p1', 'mass_p2',
        'linked_aa_p1', 'linked_aa_p2',
        'aa_len_p1', 'aa_len_p2',
        'spectrum_mz', 'spectrum_charge',
        'precursor_mz', 'precursor_charge',
        'precursor_mass', 'calc_mass', 'calc_mz',
        'start_pos_p1', 'start_pos_p2',
        'position_count_p1', 'position_count_p2',
        'protein_count_p1', 'protein_count_p2',
        'alpha_score', 'alpha_delta_score', 'beta_score',
        'total_fragments_p1', 'total_fragments_p2',
        'unique_peak_conservative_coverage_p1', 'unique_peak_conservative_coverage_p2',
        'conservative_fragsites_p1', 'conservative_fragsites_p2',
        'conservative_coverage_p1', 'conservative_coverage_p2',
    ]
    expected_mapping = {
        'match_score': 'score',
        'sequence_p1': 'sequence_p1',
        'sequence_p2': 'sequence_p2',
        'start_pos_p1': 'start_pos_p1',
        'start_pos_p2': 'start_pos_p2',
        'link_pos_p1': 'link_pos_p1',
        'link_pos_p2': 'link_pos_p2',
        'precursor_charge': 'charge',
        'protein_p1': 'protein_p1',
        'protein_p2': 'protein_p2',
        'decoy_p1': 'decoy_p1',
        'decoy_p2': 'decoy_p2',
        'fdr_group': 'fdr_group',
        'unique_peak_conservative_coverage_p1': 'coverage_p1',
        'unique_peak_conservative_coverage_p2': 'coverage_p2',
    }
    mapping = guess_column_names(columns)
    assert expected_mapping == mapping

def test_java_xifdr():
    columns = [
        'PSMID', 'run', 'scan', 'PeakListFileName', 'ScanId',
        'exp charge', 'exp m/z', 'exp mass', 'exp fractionalmass',
        'match charge', 'match mass', 'match fractionalmass',
        'Protein1', 'Name1', 'Description1', 'Decoy1',
        'Protein2', 'Name2', 'Description2', 'Decoy2',
        'PepSeq1', 'PepSeq2',
        'PepPos1', 'PepPos2',
        'PeptideLength1', 'PeptideLength2',
        'LinkPos1', 'LinkPos2',
        'ProteinLinkPos1', 'ProteinLinkPos2',
        'Charge', 'Crosslinker', 'CrosslinkerModMass',
        'MinFragments', 'P1Fragments', 'P2Fragments',
        'PeptidesWithDoublets', 'PeptidesWithStubs',
        'minPepCoverage', 'minPepCoverageAbsolute',
        'Score',
        'isDecoy', 'isTT', 'isTD', 'isDD',
        'fdrGroup', 'fdr', 'ifdr', 'PEP',
        '', 'PeptidePairFDR',
        'Protein1FDR', 'Protein2FDR',
        'LinkFDR', 'PPIFDR', 'peptide pair id',
        'link id', 'ppi id',
    ]
    expected_mapping = {
        'Score': 'score',
        'PepSeq1': 'sequence_p1',
        'PepSeq2': 'sequence_p2',
        'PepPos1': 'start_pos_p1',
        'PepPos2': 'start_pos_p2',
        'LinkPos1': 'link_pos_p1',
        'LinkPos2': 'link_pos_p2',
        'Charge': 'charge',
        'Protein1': 'protein_p1',
        'Protein2': 'protein_p2',
        'Decoy1': 'decoy_p1',
        'Decoy2': 'decoy_p2',
        'fdrGroup': 'fdr_group',
    }
    mapping = guess_column_names(columns)
    assert expected_mapping == mapping

def test_static_mapping_override():
    columns = [
        'match_score', 'sequenceA', 'sequenceB', 'precursor_charge',
        'startA', 'startB', 'linkA', 'linkB',
        'proteinA', 'proteinB', 'decoyA', 'decoyB',
        'fdr_group'
    ]
    static_mapping = {
        'sequenceA': 'sequence_p1',
        'sequenceB': 'sequence_p2',
        'startA': 'start_pos_p1',
        'startB': 'start_pos_p2',
        'linkA': 'link_pos_p1',
        'linkB': 'link_pos_p2',
        'proteinA': 'protein_p1',
        'proteinB': 'protein_p2',
        'decoyA': 'decoy_p1',
        'decoyB': 'decoy_p2',
    }

    mapping = guess_column_names(columns, static_mapping=static_mapping)

    expected_mapping = {
        'sequenceA': 'sequence_p1',
        'sequenceB': 'sequence_p2',
        'startA': 'start_pos_p1',
        'startB': 'start_pos_p2',
        'linkA': 'link_pos_p1',
        'linkB': 'link_pos_p2',
        'proteinA': 'protein_p1',
        'proteinB': 'protein_p2',
        'decoyA': 'decoy_p1',
        'decoyB': 'decoy_p2',
        'match_score': 'score',
        'precursor_charge': 'charge',
        'fdr_group': 'fdr_group',
    }

    assert mapping == expected_mapping
