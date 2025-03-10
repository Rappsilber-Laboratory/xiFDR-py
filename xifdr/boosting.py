import os
from functools import partial
import logging

import numpy as np
from polars import col
import polars as pl
from scipy.optimize import  brute
from multiprocessing import get_context
from .fdr import full_fdr

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
    else:
        raise ValueError(f'Unkown boosting method: {method}')

def boost_rec_brute(df: pl.DataFrame,
                    psm_fdr: (float, float) = (0.0, 1.0),
                    pep_fdr: (float, float) = (0.0, 1.0),
                    prot_fdr: (float, float) = (0.0, 1.0),
                    link_fdr: (float, float) = (0.0, 1.0),
                    ppi_fdr: (float, float) = (0.0, 1.0),
                    boost_level: str = "ppi",
                    boost_between: bool = True,
                    countdown: int = 5,
                    Ns: int = 3,
                    n_jobs: int = 1):
    start_params = (
        psm_fdr,
        pep_fdr,
        prot_fdr,
        link_fdr,
        ppi_fdr
    )
    current_params = start_params
    current_spreads = np.array([
        psm_fdr[1] - psm_fdr[0],
        pep_fdr[1] - pep_fdr[0],
        prot_fdr[1] - prot_fdr[0],
        link_fdr[1] - link_fdr[0],
        ppi_fdr[1] - ppi_fdr[0],
    ])
    func = partial(
        _optimization_template,
        df=df,
        boost_level=boost_level,
        boost_between=boost_between,
    )
    best_result = -1
    current_countdown = countdown
    current_best_params = None
    with get_context("spawn").Pool(n_jobs) as pool:
        while True:
            # Find the best params for the current search space
            best_params, result = brute(
                func,
                ranges=current_params,
                Ns=Ns,
                workers=pool.map
            )
            # Make the result positive
            result *= -1
            # Check whether there was an improvement
            if result <= best_result:
                # If no improvement decrease countdown
                current_countdown -= 1
                if current_countdown == 0:
                    break
            else:
                print(f'New highscore: {result}')
                # If improvement reset countdown and update best result/params
                best_result = result
                current_best_params = best_params
                current_countdown = countdown
            # Narrow search space
            current_spreads /= 2
            # Generate next params
            next_params = ()
            for i, p in enumerate(best_params):
                # Clip params to initial values
                next_params += ((
                    max(p - current_spreads[i]/2, start_params[i][0]),
                    min(p + current_spreads[i] / 2, start_params[i][1])
                ),)
            current_params = next_params
            print(f'Next params: {next_params}')
    # Return the final best params
    return current_best_params


def _optimization_template(fdrs,
                           df: pl.DataFrame,
                           boost_level: str = "ppi",
                           boost_between: bool = True):
    result = full_fdr(df, *fdrs)[boost_level]
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
