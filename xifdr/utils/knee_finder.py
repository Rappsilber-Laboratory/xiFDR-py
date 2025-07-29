import logging

import numpy as np
import pandas as pd
import polars as pl
from kneed import KneeLocator

from xifdr.fdr import _csm_fdr, _pep_fdr, _prot_fdr, _prot_filter, _link_fdr
from xifdr.utils.column_preparation import prepare_columns


logger = logging.getLogger(__name__)

g_kwargs = None


def find_knees(df: pl.DataFrame | pd.DataFrame,
               points=200,
               poly_deg=3,
               min_len: int = 5,
               decoy_adjunct: str = 'REV_',
               unique_csm: bool = True,
               prepare_column: bool = True,
               td_prob: int = 2,
               td_prot_prob: int = 10,
               td_dd_ratio: float = 1.0,
               custom_aggs: dict = None) -> tuple[float, float, float]:
    """

    Parameters
    ----------
    df
        Input CSM dataframe
    points
        Resolution to find knees
    poly_deg
        Kneed polynome degree
    min_len
        Minimum peptide sequence length
    decoy_adjunct
        Prefix/Suffix indicating a decoy match
    unique_csm
        Make CSMs unique
    prepare_column
        Perform preparation of aggregation columns like sorting ambiguous proteins and swapping protein 1/2
    td_prob
        Minimum theoretical TD machtes for the FDR levels (except protein level)
    td_prot_prob
        Minimum theoretical TD machtes for the protein FDR level
    td_dd_ratio
        Minimum ratio of TD/DD
    custom_aggs
        Custom aggregation functions for the FDR levels

    Returns
    -------
        Return a dict with keys `'csm'`, `'pep'`, `'prot'`, `'link'`, `'ppi'` that contains the resulting polars DataFrame for each FDR level.
    """
    global g_kwargs
    aggs = {
        'pep': (pl.col('score') ** 2).sum().sqrt(),
        'prot': (pl.col('score') ** 2).sum().sqrt(),
        'link': (pl.col('score') ** 2).sum().sqrt(),
        'ppi': (pl.col('score') ** 2).sum().sqrt(),
    }
    if custom_aggs is not None:
        aggs.update(custom_aggs)

    if prepare_column:
        df = prepare_columns(df)

    # Filter CSMs for minimum peptide length
    df = df.filter(
        pl.col('sequence_p1').str.replace_all('[^A-Z]', '').str.len_chars() >= min_len,
        pl.col('sequence_p2').str.replace_all('[^A-Z]', '').str.len_chars() >= min_len,
    )

    # Check for required columns
    required_columns = [
        'score',  # Match score
        'decoy_p1', 'decoy_p2',  # Target/decoy classification
        'charge',  # Precursor charge
        'start_pos_p1', 'start_pos_p2',  # Position of peptides in proteins origins
        'link_pos_p1', 'link_pos_p2',  # Position of the link in the peptides
        'sequence_p1', 'sequence_p2',  # Peptide sequences including modifications
        'protein_p1', 'protein_p2',  # Protein origins of the peptides
    ]

    # Check for required columns
    missing_columns = [
        c for c in required_columns
        if c not in df.columns
    ]
    if len(missing_columns) > 0:
        raise Exception(f'Missing required columns: {missing_columns}')

    # Aggregate unique CSMs
    never_agg_cols = ['fdr_group', 'decoy_class', 'TT', 'TD', 'DD']
    first_aggs = [
        pl.col(c).get(0)
        for c in never_agg_cols
    ]
    never_agg_cols += ['score', 'protein_score_p1', 'protein_score_p2']

    df_csm = _csm_fdr(df, 1, unique_csm, td_prob, td_dd_ratio)
    x = np.linspace(0, 1, points, endpoint=True)
    y = np.array([
        df_csm.filter(pl.col('csm_fdr') <= _x).height
        for _x in x
    ])

    csm_knee_fdr = 0.5
    kn = KneeLocator(
        x, y,
        curve="concave",
        interp_method="polynomial",
        polynomial_degree=poly_deg
    )
    if kn.all_knees and kn.find_knee()[0] > 0:
        csm_knee_fdr = kn.find_knee()[0]

    df_csm = df_csm.filter(pl.col('csm_fdr') <= csm_knee_fdr)

    # Calculate peptide FDR and filter
    logger.debug('Calculate peptide FDR and filter')
    df_pep = _pep_fdr(df_csm, aggs['pep'], 1.0, first_aggs, never_agg_cols, td_prob, td_dd_ratio)

    x = np.linspace(0, 1, points, endpoint=True)
    y = np.array([
        df_pep.filter(pl.col('pep_fdr') <= _x).height
        for _x in x
    ])

    pep_knee_fdr = .5
    kn = KneeLocator(
        x, y,
        curve="concave",
        interp_method="polynomial",
        polynomial_degree=poly_deg
    )
    if kn.all_knees and kn.find_knee()[0] > 0:
        pep_knee_fdr = kn.find_knee()[0]

    df_pep = df_pep.filter(pl.col('pep_fdr') <= pep_knee_fdr)

    # Calculate protein FDR and filter
    logger.debug('Calculate protein FDR and filter')
    df_prot = _prot_fdr(df_pep, aggs['prot'], 1.0, td_prot_prob)
    x = np.linspace(0, 1, points, endpoint=True)
    y = np.array([
        df_prot.filter(pl.col('prot_fdr') <= _x).height
        for _x in x
    ])

    prot_knee_fdr = .5
    kn = KneeLocator(
        x, y,
        curve="concave",
        interp_method="polynomial",
        polynomial_degree=poly_deg
    )
    if kn.all_knees and kn.find_knee()[0] > 0:
        prot_knee_fdr = kn.find_knee()[0]

    df_prot = df_prot.filter(pl.col('prot_fdr') <= prot_knee_fdr)

    logger.debug('Filter peptide pairs for passed proteins')
    df_pep = _prot_filter(df_pep, df_prot, decoy_adjunct)

    # Calculate link FDR and cutoff
    logger.debug('Calculate link FDR and cutoff')
    df_link = _link_fdr(df_pep, aggs['link'], 1.0, first_aggs, never_agg_cols, td_prob, td_dd_ratio)
    x = np.linspace(0, 1, points, endpoint=True)
    y = np.array([
        df_link.filter(pl.col('link_fdr') <= _x).height
        for _x in x
    ])

    link_knee_fdr = 0.5
    kn = KneeLocator(
        x, y,
        curve="concave",
        interp_method="polynomial",
        polynomial_degree=poly_deg
    )
    if kn.all_knees and kn.find_knee()[0] > 0:
        link_knee_fdr = kn.find_knee()[0]

    return (csm_knee_fdr, pep_knee_fdr, prot_knee_fdr, link_knee_fdr)