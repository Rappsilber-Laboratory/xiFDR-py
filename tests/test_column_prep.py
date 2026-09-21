import polars as pl
from polars.testing import assert_frame_equal
from xifdr.utils.column_preparation import prepare_columns

def test_column_prep():
    df = pl.DataFrame([
        [['C'], ['E', 'A'], ['A', 'B', 'B'], ['A']],  # protein_p1
        [[1],   [2,   1],   [1,   3,   2], [1]],    # start_p1
        [['A'], ['E', 'E', 'E'], ['F', 'X'], ['A']],  # protein_p2
        [[1],   [2,   1,   1],   [2, 1], [10]],      # start_p2
        [1, 2, 3, 1],  # link_pos_p1
        [7, 8, 9, 1],  # link_pos_p2
        ['ABC', 'D[MOD]EF', 'GmodHI', 'ABC'],  # sequence_p1
        ['AB{MOD}C', 'AAA', 'D(MOD)EF', 'XYZ'],  # sequence_p2
        [False, False, True, False],  # decoy_p1
        [False, True, True, False],  # decoy_p2
        [-1, 0, 1, -1],  # score
    ], schema=[
        "protein_p1",
        "start_pos_p1",
        "protein_p2",
        "start_pos_p2",
        "link_pos_p1",
        "link_pos_p2",
        "sequence_p1",
        "sequence_p2",
        "decoy_p1",
        "decoy_p2",
        "score",
    ])

    df_expect = pl.DataFrame({
        "protein_p1": [['A'], ['A', 'E'], ['A', 'B', 'B'], ['A']],
        "ppi_protein_p1": [['A'], ['A', 'E'], ['A', 'B'], ['A']],
        "start_pos_p1": [[1], [1, 2], [1, 2, 3], [1]],
        "protein_p2": [['C'], ['E', 'E'], ['F', 'X'], ['A']],
        "ppi_protein_p2": [['C'], ['E'], ['F', 'X'], ['A']],
        "start_pos_p2": [[1], [1, 2], [2, 1], [10]],
        "link_pos_p1": [7, 2, 3, 1],
        "link_pos_p2": [1, 8, 9, 1],
        "sequence_p1": ['AB{MOD}C', 'D[MOD]EF', 'GmodHI', 'ABC'],
        "sequence_p2": ['ABC', 'AAA', 'D(MOD)EF', 'XYZ'],
        "base_sequence_p1": ['ABC', 'DEF', 'GHI', 'ABC'],
        "base_sequence_p2": ['ABC', 'AAA', 'DEF', 'XYZ'],
        "decoy_p1": [False, False, True, False],
        "decoy_p2": [False, True, True, False],
        "score": [0.2, 1.2, 2.2, 0.2],
        "fdr_group": ['between', 'overlapping', 'between', 'self'],
        "decoy_class": ['TT', 'TD', 'DD', 'TT'],
        "cl_pos_p1": [[7], [2, 3], [3, 4, 5], [1]],
        "cl_pos_p2": [[1], [8, 9], [10, 9], [10]],
        "TT": [True, False, False, True],
        "TD": [False, True, False, False],
        "DD": [False, False, True, False],
        "coverage_p1": [0.5, 0.5, 0.5, 0.5],
        "coverage_p2": [0.5, 0.5, 0.5, 0.5],
        "protein_score_p1": [0.1, 0.6, 1.1, 0.1],
        "protein_score_p2": [0.1, 0.6, 1.1, 0.1],
        "group_swapped": [True, False, False, False],
    })

    df_res = prepare_columns(df)
    assert_frame_equal(
        df_res.select(df_expect.columns).sort(df_expect.columns),
        df_expect.sort(df_expect.columns)
    )
    # Check that the resulting state is constant
    df_res = prepare_columns(df)
    assert_frame_equal(
        df_res.sort(df_expect.columns),
        df_expect.select(df_res.columns).sort(df_expect.columns)
    )
