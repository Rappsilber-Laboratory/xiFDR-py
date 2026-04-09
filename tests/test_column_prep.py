import polars as pl
from polars.testing import assert_frame_equal
from xifdr.utils.column_preparation import prepare_columns

def test_column_prep():
    df = pl.DataFrame([
        [['C'], ['E', 'A'], ['A', 'B', 'B']],  # protein_p1
        [[1],   [2,   1],   [1,   3,   2]],    # start_p1
        [['A'], ['E', 'E', 'E'], ['F', 'X']],  # protein_p2
        [[1],   [2,   1,   1],   [2, 1]],      # start_p2
        [1, 2, 3],  # link_pos_p1
        [7, 8, 9],  # link_pos_p2
        ['ABC', 'DEF', 'GHI'],  # sequence_p1
        ['ABC', 'AAA', 'DEF'],  # sequence_p2
        [False, False, True],  # decoy_p1
        [False, True, True],  # decoy_p2
        [-1, 0, 1],  # score
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
        "protein_p1": [['A'], ['A', 'E'], ['A', 'B', 'B']],
        "start_pos_p1": [[1], [1, 2], [1, 2, 3]],
        "protein_p2": [['C'], ['E', 'E', 'E'], ['F', 'X']],
        "start_pos_p2": [[1], [1, 1, 2], [2, 1]],
        "link_pos_p1": [1, 2, 3],
        "link_pos_p2": [7, 8, 9],
        "sequence_p1": ['ABC', 'DEF', 'GHI'],
        "sequence_p2": ['ABC', 'AAA', 'DEF'],
        "decoy_p1": [False, False, True],
        "decoy_p2": [False, True, True],
        "score": [0.2, 1.2, 2.2],
        "fdr_group": ['between', 'self', 'between'],
        "decoy_class": ['TT', 'TD', 'DD'],
        "cl_pos_p1": [[1], [2, 3], [3, 4, 5]],
        "cl_pos_p2": [[7], [8, 8, 9], [10, 9]],
        "TT": [True, False, False],
        "TD": [False, True, False],
        "DD": [False, False, True],
        "coverage_p1": [0.5, 0.5, 0.5],
        "coverage_p2": [0.5, 0.5, 0.5],
        "protein_score_p1": [0.1, 0.6, 1.1],
        "protein_score_p2": [0.1, 0.6, 1.1]
    })

    df_res = prepare_columns(df)
    assert_frame_equal(df_res, df_expect.select(df_res.columns))
