import polars as pl
from xifdr.utils.column_preparation import prepare_columns

def test_column_prep():
    df = pl.DataFrame([
        [['C'], ['E', 'A'], ['A', 'B', 'B']],  # protein_p1
        [[1],   [2,   1],   [1,   3,   2]],    # start_p1
        [['A'], ['E', 'E', 'E'], ['X', 'F']],  # protein_p2
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

    df_expect = pl.DataFrame([
        [['A'], ['A', 'E'], ['F', 'X']],  # protein_p1
        [[1], [1, 2], [1, 2]],  # start_p1
        [['C'], ['E', 'E', 'E'], ['A', 'B', 'B']],  # protein_p2
        [[1], [1, 1, 2], [1, 2, 3]],  # start_p2
        [7, 2, 9],  # link_pos_p1
        [1, 8, 3],  # link_pos_p2
        ['ABC', 'DEF', 'DEF'],  # sequence_p1
        ['ABC', 'AAA', 'GHI'],  # sequence_p2
        [False, False, True],  # decoy_p1
        [False, True, True],  # decoy_p2
        [0, 1, 2],  # score
        ['TT', 'TD', 'DD'],  # decoy_class
        [[7], [2, 3], [9, 10]],  # cl_pos_p1
        [[1], [8, 8, 9], [3, 4, 5]],  # cl_pos_p2
        [0.0, .5, 1],  #  protein_score_p1
        [0.0, .5, 1],  #  protein_score_p2
        [.5, .5, .5],  # coverage_p1
        [.5, .5, .5],  # coverage_p2
        [True, False, False],  # TT
        [False, True, False],  # TD
        [False, False, True],  # DD
        ['between', 'self', 'between'], # fdr_group
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
        "decoy_class",
        "cl_pos_p1",
        "cl_pos_p2",
        "protein_score_p1",
        "protein_score_p2",
        "coverage_p1",
        "coverage_p2",
        "TT", "TD", "DD",
        "fdr_group"
    ])

    df_res = prepare_columns(df)
    assert(
        (df_res == df_expect.select(df_res.columns)).to_numpy().all()
    )
