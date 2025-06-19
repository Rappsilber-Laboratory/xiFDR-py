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