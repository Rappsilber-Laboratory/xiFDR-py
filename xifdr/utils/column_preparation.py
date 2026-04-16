import numpy as np
import polars as pl
from datetime import datetime

def prepare_columns(df, decoy_adjunct:str = 'REV_'):
    """Prepares and processes a Polars DataFrame for protein-protein interaction analysis.

    This function ensures the proper format of protein columns, calculates crosslink positions,
    sorts protein lists, swaps peptides based on predefined criteria, and computes various scores
    and classification labels.

    Parameters
    ----------
    df : pl.DataFrame
        A Polars DataFrame containing protein interaction data. If not already a Polars DataFrame,
        it will be converted.

    Returns
    -------
    pl.DataFrame
        The processed DataFrame with formatted columns, calculated positions, sorted lists,
        swapped peptides, and additional computed fields.

    Notes
    -----
    - Converts semicolon-separated protein columns into lists.
    - Computes crosslink positions by adjusting start positions.
    - Sorts list columns based on protein group and start position order.
    - Swaps peptides based on a custom string comparison mask.
    - Computes one-hot encoded labels (`TT`, `TD`, `DD`) for classification.
    - Ensures a positive score and assigns dummy coverage values if missing.
    - Computes proportional protein scores based on coverage.
    """
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
        if not df[c].dtype.is_nested():
            df = df.with_columns(
                pl.col(c).cast(pl.String).str.replace_all(
                    '[ ]*', ''  # Remove all spaces
                ).str.split(';')
            )

    # Generate fdr_group if not present
    if 'fdr_group' not in df.columns:
        df = df.with_columns(
            fdr_group=(
                (pl.col('protein_p1').list.eval(pl.element().str.replace(decoy_adjunct, '')).list.set_intersection(
                    pl.col('protein_p2').list.eval(pl.element().str.replace(decoy_adjunct, ''))
                ).list.len()==0).cast(pl.String).replace(
                    ['true', 'false'],
                    ['between', 'self']
                )
            )
        ).with_columns(
            fdr_group=pl.when(pl.col('protein_p2').eq([]) | pl.col('protein_p2').is_null()).then(
                pl.lit('linear')
            ).otherwise(
                pl.col('fdr_group')
            )
        )

    # Create decoy_class column if not present
    if 'decoy_class' not in df.columns:
        df = df.with_columns(
            decoy_class=pl.when(
                pl.col('decoy_p1') & pl.col('decoy_p2')
            ).then(
                pl.lit('DD')
            ).when(
                pl.col('decoy_p1') ^ pl.col('decoy_p2')
            ).then(
                pl.lit('TD')
            ).otherwise(
                pl.lit('TT')
            )
        )

    # Sort list columns by protein group order
    df = df.with_columns(
        pl.col(list_cols_1).fill_null(pl.lit([])),
        pl.col(list_cols_2).fill_null(pl.lit([])),
    ).with_columns(
        _tmp_join = pl.int_range(pl.len())  # Id range to reverse the explode
    ).explode(list_cols_1).sort(
        pl.col('protein_p1'),
        pl.col('start_pos_p1')
    ).group_by('_tmp_join', maintain_order=True).agg(
        pl.col(list_cols_1).drop_nulls(),
        pl.exclude(list_cols_1).first(),
    ).explode(list_cols_2).sort(
        pl.col('protein_p2'),
        pl.col('start_pos_p2')
    ).group_by('_tmp_join', maintain_order=True).agg(
        pl.col(list_cols_2).drop_nulls(),
        pl.exclude(list_cols_2).first(),
    )
    #df = df.with_columns(
    #    pl.col('protein_p2').fill_null([]),
    #    pl.col('start_pos_p2').fill_null([]),
    #).with_columns(
    #    multi_list_sort(*list_cols_1)
    #).with_columns(
    #    multi_list_sort(*list_cols_2)
    #)

    # Calculate crosslink position in protein
    df = df.with_columns(
        cl_pos_p1 = pl.col('start_pos_p1').cast(pl.List(pl.Int64)) + pl.col('link_pos_p1') - 1,
        cl_pos_p2 = pl.col('start_pos_p2').cast(pl.List(pl.Int64)) + pl.col('link_pos_p2') - 1,
    )

    # Put in dummy coverage if none provided
    if 'coverage_p1' not in df.columns or 'coverage_p2' not in df.columns:
        df = df.with_columns(
            coverage_p1 = pl.lit(0.5),
            coverage_p2 = pl.lit(0.5),
        )

    coverage_p1_prop = pl.col('coverage_p1') / (pl.col('coverage_p1') + pl.col('coverage_p2'))
    coverage_p2_prop = pl.col('coverage_p2') / (pl.col('coverage_p1') + pl.col('coverage_p2'))
    df = df.with_columns(
        protein_score_p1 = pl.col('score') * coverage_p1_prop,
        protein_score_p2 = pl.col('score') * coverage_p2_prop
    )

    # Swap peptides based on joined protein group
    df = df.with_columns(
        pl.col(['protein_p1', 'protein_p2', 'cl_pos_p1', 'cl_pos_p2']).name.prefix('_tmp_respair_swap_'),
        _tmp_join = pl.int_range(pl.len()),
    ).explode('_tmp_respair_swap_protein_p1', '_tmp_respair_swap_cl_pos_p1').unique(
        subset=['_tmp_join', '_tmp_respair_swap_protein_p1', '_tmp_respair_swap_cl_pos_p1']
    ).group_by('_tmp_join').agg(
        pl.col(['_tmp_respair_swap_protein_p1', '_tmp_respair_swap_cl_pos_p1']).drop_nulls(),
        pl.exclude(['_tmp_respair_swap_protein_p1', '_tmp_respair_swap_cl_pos_p1']).first()
    ).explode('_tmp_respair_swap_protein_p2', '_tmp_respair_swap_cl_pos_p2').unique(
        subset=['_tmp_join', '_tmp_respair_swap_protein_p2', '_tmp_respair_swap_cl_pos_p2']
    ).group_by('_tmp_join').agg(
        pl.col(['_tmp_respair_swap_protein_p2', '_tmp_respair_swap_cl_pos_p2']).drop_nulls(),
        pl.exclude(['_tmp_respair_swap_protein_p2', '_tmp_respair_swap_cl_pos_p2']).first()
    ).drop('_tmp_join')

    df = df.with_columns(
        pl.col(['protein_p1', 'protein_p2']).list.unique().list.sort().name.prefix('_tmp_ppi_swap_'),
    )

    swap_cmp_cols = [
        (f"{c}_p1", f"{c}_p2")
        for c in [
            '_tmp_ppi_swap_protein',
            '_tmp_respair_swap_protein',
            '_tmp_respair_swap_cl_pos',
            'protein',
            'cl_pos',
            'link_pos',
            'sequence',
        ]
    ]

    # Generate swap mask
    swap_cond = pl.when(pl.lit(False)).then(pl.lit(None))
    for c1, c2 in swap_cmp_cols:
        c21_list = pl.concat_list([c2, c1])
        c12_list = pl.concat_list([c1, c2])
        swap_cond = swap_cond.when(
            # Test if columns need swapping
            c12_list != c12_list.list.sort()
        ).then(pl.lit(True))
        swap_cond = swap_cond.when(
            # Test if columns differ and we can stop here
            c12_list != c21_list
        ).then(pl.lit(False))
        # If columns are equal we need to continue comparing
    swap_cond = swap_cond.otherwise(pl.lit(False))

    # Swap peptide specific columns
    pair_cols1 = ['sequence_p1', 'protein_p1', 'start_pos_p1', 'link_pos_p1', 'cl_pos_p1', 'coverage_p1', 'decoy_p1']
    pair_cols2 = ['sequence_p2', 'protein_p2', 'start_pos_p2', 'link_pos_p2', 'cl_pos_p2', 'coverage_p2', 'decoy_p2']

    for c1, c2 in zip(pair_cols1, pair_cols2):
        df = df.with_columns(
           pl.when(swap_cond).then(pl.col(c2)).otherwise(pl.col(c1)).alias(c1),
           pl.when(swap_cond).then(pl.col(c1)).otherwise(pl.col(c2)).alias(c2),
        )
    
    df = df.drop(pl.selectors.matches('^_tmp_respair_swap_'))

    # Calculate one-hot encoded target/decoy labels
    df = df.with_columns(
        TT=(pl.col('decoy_class')=='TT'),
        TD=(pl.col('decoy_class')=='TD'),
        DD=(pl.col('decoy_class')=='DD'),
    )

    # Fill in infinite scores
    max_score = df.filter(pl.col('score') < np.inf)['score'].max()
    min_score = df.filter(pl.col('score') > -np.inf)['score'].min()
    inf_margin = (max_score-min_score)*0.1
    df = df.with_columns(pl.col('score') - min_score + inf_margin)
    df = df.with_columns(
        score=pl.when(pl.col('score') == np.inf).then(
            pl.lit(max_score) + 2*pl.lit(inf_margin)
        ).when(pl.col('score') == -np.inf).then(
            pl.lit(0)
        ).otherwise(
            pl.col('score')
        )
    )

    df = df.with_columns(
        pl.col(list_cols_1).replace([], None),
        pl.col(list_cols_2).replace([], None),
    )

    return df
