import polars as pl
from polars import col
from .double_argsort_batch import double_argsort_batch


def prepare_columns(df):
    if not isinstance(df, pl.DataFrame):
        df: pl.DataFrame = pl.DataFrame(df)

    # Convert semicolon separated string columns to lists
    list_cols_1 = [
        'protein_p1', 'start_pos_p1'
    ]
    list_cols_2 = [
        'protein_p2', 'start_pos_p2'
    ]
    list_cols = list_cols_1 + list_cols_2
    for c in list_cols:
        if not isinstance(df[c].dtype, pl.List):
            df = df.with_columns(
                col(c).cast(pl.String).str.split(';')
            )

    # Sort list columns by protein group order
    protein_p1_ord = pl.struct(["protein_p1", "start_pos_p1"]).map_batches(
        lambda x: pl.Series(double_argsort_batch(
            x.struct.field('protein_p1').to_list(),
            x.struct.field('start_pos_p1').to_list()
        ))
    )
    protein_p2_ord = pl.struct(["protein_p2", "start_pos_p2"]).map_batches(
        lambda x: pl.Series(double_argsort_batch(
            x.struct.field('protein_p2').to_list(),
            x.struct.field('start_pos_p2').to_list()
        ))
    )
    df = df.with_columns(
        protein_p1_ord = protein_p1_ord
    )
    df = df.with_columns(
        col('protein_p2').fill_null([])
    ).with_columns(
        protein_p2_ord = protein_p2_ord
    )

    for c in list_cols_1:
        df = df.with_columns(
            col(c).list.gather(col('protein_p1_ord'))
        )
    for c in list_cols_2:
        df = df.with_columns(
            col(c).list.gather(col('protein_p2_ord'))
        )

    # Swap peptides based on joined protein group
    swap_mask = df.select(
        (
            col('protein_p1').list.join(';')+'§'+
            col('start_pos_p1').cast(pl.List(pl.String)).list.join(';')+'§'+
            col('link_pos_p1').cast(pl.List(pl.String)).list.join(';')+'§'+
            col('sequence_p1')
        ) > (
            col('protein_p2').list.join(';')+'§'+
            col('start_pos_p2').cast(pl.List(pl.String)).list.join(';')+'§'+
            col('link_pos_p2').cast(pl.List(pl.String)).list.join(';')+'§'+
            col('sequence_p2')
        )
    ).to_series()
    pair_cols1 = ['sequence_p1', 'protein_p1', 'start_pos_p1', 'link_pos_p1', 'decoy_p1']
    pair_cols2 = ['sequence_p2', 'protein_p2', 'start_pos_p2', 'link_pos_p2', 'decoy_p2']
    for c1, c2 in zip(pair_cols1, pair_cols2):
        df = df.with_columns(
           pl.when(swap_mask).then(col(c2)).otherwise(col(c1)).alias(c1),
           pl.when(swap_mask).then(col(c1)).otherwise(col(c2)).alias(c2),
        )

    # Calculate one-hot encoded target/decoy labels
    df = df.with_columns(
        TT=(pl.col('decoy_class')=='TT'),
        TD=(pl.col('decoy_class')=='TD'),
        DD=(pl.col('decoy_class')=='DD'),
    )

    # Calculate decoy_class column
    df = df.with_columns(
        decoy_class = pl.when(
            col('decoy_p1') & (col('decoy_p2') | col('decoy_p2').is_null())
        ).then(pl.lit('DD')).when(
            (~col('decoy_p1')) & ((~col('decoy_p2')) | col('decoy_p2').is_null())
        ).then(pl.lit('TT')).otherwise(pl.lit('TD'))
    )

    # Make score positive
    df.with_columns(col('score') + col('score').min())

    # Put in dummy coverage if none provided
    if 'coverage_p1' not in df.columns or 'coverage_p2' not in df.columns:
        df = df.with_columns(
            coverage_p1 = pl.lit(0.5),
            coverage_p2 = pl.lit(0.5)
        )

    coverage_p1_prop = col('coverage_p1') / (col('coverage_p1') + col('coverage_p2'))
    coverage_p2_prop = col('coverage_p2') / (col('coverage_p1') + col('coverage_p2'))
    df = df.with_columns(
        protein_score_p1 = col('score') * coverage_p1_prop,
        protein_score_p2 = col('score') * coverage_p2_prop
    )

    # Calculate crosslink position in protein
    df = df.with_columns(
        cl_pos_p1 = col('start_pos_p1').cast(pl.List(pl.Int64)) + col('link_pos_p1'),
        cl_pos_p2 = col('start_pos_p2').cast(pl.List(pl.Int64)) + col('link_pos_p2'),
    )

    return df