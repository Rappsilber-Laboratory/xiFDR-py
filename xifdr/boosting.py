import os
from functools import partial
import logging

import numpy as np
from polars import col
import polars as pl
from scipy.optimize import  brute
from multiprocessing import get_context
from .fdr import full_fdr
from .utils.column_preparation import prepare_columns
from .optimization import manhattan, independent_gird

logger = logging.getLogger(__name__)

def boost(df: pl.DataFrame,
          psm_fdr: (float, float) = (0.0, 1.0),
          pep_fdr: (float, float) = (0.0, 1.0),
          prot_fdr: (float, float) = (0.0, 1.0),
          link_fdr: (float, float) = (0.0, 1.0),
          ppi_fdr: (float, float) = (0.0, 1.0),
          boost_level: str = "ppi",
          boost_between: bool = True,
          method: str = "brute",
          points: int = 5,
          n_jobs: int = 1):
    if method == 'brute':
        return boost_rec_brute(
            df=df,
            psm_fdr=psm_fdr,
            pep_fdr=pep_fdr,
            prot_fdr=prot_fdr,
            link_fdr=link_fdr,
            ppi_fdr=ppi_fdr,
            boost_level=boost_level,
            boost_between=boost_between,
            n_jobs=n_jobs
        )
    elif method == 'manhattan':
        return boost_manhattan(
            df=df,
            psm_fdr=psm_fdr,
            pep_fdr=pep_fdr,
            prot_fdr=prot_fdr,
            link_fdr=link_fdr,
            ppi_fdr=ppi_fdr,
            boost_level=boost_level,
            boost_between=boost_between,
            points=points,
            n_jobs=n_jobs
        )
    elif method == 'independent_grid':
        return boost_independent_grid(
            df=df,
            psm_fdr=psm_fdr,
            pep_fdr=pep_fdr,
            prot_fdr=prot_fdr,
            link_fdr=link_fdr,
            ppi_fdr=ppi_fdr,
            boost_level=boost_level,
            boost_between=boost_between,
            points=points,
            n_jobs=n_jobs
        )
    else:
        raise ValueError(f'Unkown boosting method: {method}')

def boost_manhattan(df: pl.DataFrame,
                    psm_fdr: (float, float) = (0.0, 1.0),
                    pep_fdr: (float, float) = (0.0, 1.0),
                    prot_fdr: (float, float) = (0.0, 1.0),
                    link_fdr: (float, float) = (0.0, 1.0),
                    ppi_fdr: (float, float) = (0.0, 1.0),
                    boost_level: str = "ppi",
                    boost_between: bool = True,
                    points: int = 3,
                    n_jobs: int = 1):
    df = prepare_columns(df)
    start_params = (
        psm_fdr,
        pep_fdr,
        prot_fdr,
        link_fdr,
        ppi_fdr
    )
    with get_context('forkserver').Pool(n_jobs) as pool:
        best_params, result = manhattan(
            _optimization_template,
            kwargs=dict(
                df=df,
                boost_level = boost_level,
                boost_between=boost_between,
            ),
            ranges=start_params,
            points=points,
            workers=pool.map,
        )
    return best_params


def boost_independent_grid(df: pl.DataFrame,
                    psm_fdr: (float, float) = (0.0, 1.0),
                    pep_fdr: (float, float) = (0.0, 1.0),
                    prot_fdr: (float, float) = (0.0, 1.0),
                    link_fdr: (float, float) = (0.0, 1.0),
                    ppi_fdr: (float, float) = (0.0, 1.0),
                    boost_level: str = "ppi",
                    boost_between: bool = True,
                    points: int = 3,
                    n_jobs: int = 1):
    df = prepare_columns(df)
    start_params = (
        psm_fdr,
        pep_fdr,
        prot_fdr,
        link_fdr,
        ppi_fdr
    )
    with get_context('forkserver').Pool(n_jobs) as pool:
        best_params, result = independent_gird(
            _optimization_template,
            kwargs=dict(
                df=df,
                boost_level = boost_level,
                boost_between=boost_between,
            ),
            ranges=start_params,
            points=points,
            workers=pool.map,
        )
    return best_params


def boost_rec_brute(df: pl.DataFrame,
                    psm_fdr: (float, float) = (0.0, 1.0),
                    pep_fdr: (float, float) = (0.0, 1.0),
                    prot_fdr: (float, float) = (0.0, 1.0),
                    link_fdr: (float, float) = (0.0, 1.0),
                    ppi_fdr: (float, float) = (0.0, 1.0),
                    boost_level: str = "ppi",
                    boost_between: bool = True,
                    Ns: int = 3,
                    n_jobs: int = 1):
    df = prepare_columns(df)
    start_params = (
        psm_fdr,
        pep_fdr,
        prot_fdr,
        link_fdr,
        ppi_fdr
    )
    func = partial(
        _optimization_template,
        df=df,
        boost_level=boost_level,
        boost_between=boost_between,
    )
    with get_context("spawn").Pool(n_jobs) as pool:
        best_params, result = manhattan(
            func,
            ranges=start_params,
            points=5,
            workers=pool.map,
        )
    return best_params


def _optimization_template(fdrs,
                           df: pl.DataFrame,
                           boost_level: str = "ppi",
                           boost_between: bool = True):
    result = full_fdr(df, *fdrs, prepare_column=False)[boost_level]
    if boost_between:
        result = result.filter(col('fdr_group') == 'between')
    tt = len(result.filter(col('TT')))
    td = len(result.filter(col('TD')))
    dd = len(result.filter(col('DD')))
    tp = tt + td - dd
    print(
        f'Estimated true positive matches: {tp}\n'
        f'Parameters: {fdrs}'
    )
    return -tp
