from functools import partial
import logging

from polars import col
import polars as pl
from contextlib import closing
from multiprocessing import get_context
from .fdr import full_fdr
from .utils.column_preparation import prepare_columns
from .optimization import manhattan, independent_gird

logger = logging.getLogger(__name__)

def boost(df: pl.DataFrame,
          csm_fdr: (float, float) = (0.0, 1.0),
          pep_fdr: (float, float) = (0.0, 1.0),
          prot_fdr: (float, float) = (0.0, 1.0),
          link_fdr: (float, float) = (0.0, 1.0),
          ppi_fdr: (float, float) = (0.0, 1.0),
          min_len: int = 5,
          unique_csm: bool = True,
          boost_cols: list = [],
          neg_boost_cols: list = [],
          boost_level: str = "ppi",
          boost_between: bool = True,
          td_prob: int = 2,
          td_prot_prob: int = 10,
          method: str = "manhattan",
          countdown: int = 3,
          points: int = 5,
          n_jobs: int = 1) -> (float, float, float, float, float):
    """
    Find the best FDR cutoffs to optimize results for a certain FDR level.

    Parameters
    ----------
    df
        CSM DataFrame
    csm_fdr
        Search range for CSM FDR level cutoff
    pep_fdr
        Search range for peptide FDR level cutoff
    prot_fdr
        Search range for protein FDR level cutoff
    link_fdr
        Search range for residue link FDR level cutoff
    ppi_fdr
        Search range for protein pair FDR level cutoff
    min_len
        Minimum peptide sequence length
    boost_level
        FDR level tp boost for
    boost_between
        Whether to boost for between links
    td_prob
        Minimum theoretical TD machtes for the FDR levels (except protein level)
    td_prot_prob
        Minimum theoretical TD machtes for the protein FDR level
    method
        Search algorithm to use
    countdown
        Number interation without improvement to stop
    points
        Number of FDR cutoffs to search in one iteration
    n_jobs
        Number of threads to use

    Returns
    -------
        Returns a tuple with the optimal FDR levels.
    """
    if method == 'brute':
        return boost_rec_brute(
            df=df,
            csm_fdr=csm_fdr,
            pep_fdr=pep_fdr,
            prot_fdr=prot_fdr,
            link_fdr=link_fdr,
            ppi_fdr=ppi_fdr,
            min_len=min_len,
            unique_csm=unique_csm,
            boost_level=boost_level,
            boost_between=boost_between,
            n_jobs=n_jobs
        )
    elif method == 'manhattan':
        return boost_manhattan(
            df=df,
            csm_fdr=csm_fdr,
            pep_fdr=pep_fdr,
            prot_fdr=prot_fdr,
            link_fdr=link_fdr,
            ppi_fdr=ppi_fdr,
            min_len=min_len,
            unique_csm=unique_csm,
            boost_cols=boost_cols,
            neg_boost_cols=neg_boost_cols,
            boost_level=boost_level,
            boost_between=boost_between,
            td_prob=td_prob,
            td_prot_prob=td_prot_prob,
            countdown=countdown,
            points=points,
            n_jobs=n_jobs
        )
    elif method == 'independent_grid':
        return boost_independent_grid(
            df=df,
            csm_fdr=csm_fdr,
            pep_fdr=pep_fdr,
            prot_fdr=prot_fdr,
            link_fdr=link_fdr,
            ppi_fdr=ppi_fdr,
            min_len=min_len,
            unique_csm=unique_csm,
            boost_level=boost_level,
            boost_between=boost_between,
            points=points,
            n_jobs=n_jobs
        )
    else:
        raise ValueError(f'Unkown boosting method: {method}')

def boost_manhattan(df: pl.DataFrame,
                    csm_fdr: (float, float) = (0.0, 1.0),
                    pep_fdr: (float, float) = (0.0, 1.0),
                    prot_fdr: (float, float) = (0.0, 1.0),
                    link_fdr: (float, float) = (0.0, 1.0),
                    ppi_fdr: (float, float) = (0.0, 1.0),
                    min_len: int = 5,
                    unique_csm: bool = True,
                    boost_cols: list = [],
                    neg_boost_cols: list = [],
                    boost_level: str = "ppi",
                    boost_between: bool = True,
                    td_prob: int = 2,
                    td_prot_prob: int = 10,
                    countdown: int = 3,
                    points: int = 3,
                    n_jobs: int = 1):
    df = prepare_columns(df)
    start_params = (
        csm_fdr,
        pep_fdr,
        prot_fdr,
        link_fdr,
        ppi_fdr
    )
    start_params += tuple((0.0, 1.0) for _ in boost_cols)
    start_params += tuple((0.0, 1.0) for _ in neg_boost_cols)

    with closing(get_context('spawn').Pool(n_jobs)) as pool:
        best_params, result = manhattan(
            _optimization_template,
            kwargs=dict(
                df=df,
                min_len=min_len,
                unique_csm=unique_csm,
                boost_cols=boost_cols,
                neg_boost_cols=neg_boost_cols,
                boost_level=boost_level,
                boost_between=boost_between,
                td_prob=td_prob,
                td_prot_prob=td_prot_prob,
            ),
            ranges=start_params,
            countdown=countdown,
            points=points,
            workers=pool.map,
        )
    return best_params


def boost_independent_grid(df: pl.DataFrame,
                           csm_fdr: (float, float) = (0.0, 1.0),
                           pep_fdr: (float, float) = (0.0, 1.0),
                           prot_fdr: (float, float) = (0.0, 1.0),
                           link_fdr: (float, float) = (0.0, 1.0),
                           ppi_fdr: (float, float) = (0.0, 1.0),
                           min_len: int = 5,
                           unique_csm: bool = True,
                           boost_level: str = "ppi",
                           boost_between: bool = True,
                           points: int = 3,
                           n_jobs: int = 1) -> object:
    df = prepare_columns(df)
    start_params = (
        csm_fdr,
        pep_fdr,
        prot_fdr,
        link_fdr,
        ppi_fdr
    )
    with closing(get_context('spawn').Pool(n_jobs)) as pool:
        best_params, result = independent_gird(
            _optimization_template,
            kwargs=dict(
                df=df,
                boost_level=boost_level,
                boost_between=boost_between,
                min_len=min_len,
                unique_csm=unique_csm,
            ),
            ranges=start_params,
            points=points,
            workers=pool.map,
        )
    return best_params


def boost_rec_brute(df: pl.DataFrame,
                    csm_fdr: (float, float) = (0.0, 1.0),
                    pep_fdr: (float, float) = (0.0, 1.0),
                    prot_fdr: (float, float) = (0.0, 1.0),
                    link_fdr: (float, float) = (0.0, 1.0),
                    ppi_fdr: (float, float) = (0.0, 1.0),
                    min_len: int = 5,
                    unique_csm: bool = True,
                    boost_level: str = "ppi",
                    boost_between: bool = True,
                    Ns: int = 3,
                    n_jobs: int = 1):
    df = prepare_columns(df)
    start_params = (
        csm_fdr,
        pep_fdr,
        prot_fdr,
        link_fdr,
        ppi_fdr
    )
    func = partial(
        _optimization_template,
        df=df,
        min_len=min_len,
        unique_csm=unique_csm,
        boost_level=boost_level,
        boost_between=boost_between,
    )
    with closing(get_context("spawn").Pool(n_jobs)) as pool:
        best_params, result = manhattan(
            func,
            ranges=start_params,
            points=5,
            workers=pool.map,
        )
    return best_params


def _optimization_template(cutoffs,
                           df: pl.DataFrame,
                           min_len: int = 5,
                           unique_csm: bool = True,
                           boost_cols: list = [],
                           neg_boost_cols: list = [],
                           boost_level: str = "ppi",
                           boost_between: bool = True,
                           td_prob: int = 2,
                           td_prot_prob: int = 10):
    fdrs = cutoffs[:5]
    col_levels = cutoffs[5:]
    neg_col_levels = col_levels[len(boost_cols):]

    for i, c in enumerate(boost_cols):
        df = df.filter(
            (
                    (pl.col(c)-pl.col(c).min()) /
                    (pl.col(c).max()-pl.col(c).min())
            ) >= col_levels[i]
        )

    for i, c in enumerate(neg_boost_cols):
        df = df.filter(
            (
                    (pl.col(c)-pl.col(c).min()) /
                    (pl.col(c).max()-pl.col(c).min())
            ) <= neg_col_levels[i]
        )
    result = full_fdr(
        df, *fdrs,
        min_len=min_len,
        unique_csm=unique_csm,
        prepare_column=False,
        td_prob=td_prob,
        td_prot_prob=td_prot_prob
    )[boost_level]
    if boost_between:
        result = result.filter(col('fdr_group') == 'between')
    tt = len(result.filter(col('TT')))
    td = len(result.filter(col('TD')))
    dd = len(result.filter(col('DD')))
    tp = tt - td + dd
    logger.debug(
        f'Estimated true positive matches: {tp}\n'
        f'Parameters: {cutoffs}'
    )
    return -tp
