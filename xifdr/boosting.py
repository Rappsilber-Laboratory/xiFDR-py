import os
from functools import partial
import logging

import pandas as pd
from polars import col
import polars as pl
from scipy.optimize import differential_evolution
import psutil
from .fdr import full_fdr

logger = logging.getLogger(__name__)

def boost(df: pl.DataFrame,
          psm_fdr: (float, float) = (0.0, 1.0),
          pep_fdr: (float, float) = (0.0, 1.0),
          prot_fdr: (float, float) = (0.0, 1.0),
          link_fdr: (float, float) = (0.0, 1.0),
          ppi_fdr: (float, float) = (0.0, 1.0),
          boost_level: str = "ppi",
          boost_between: bool = True):
    func = partial(
        _optimization_template,
        df=df,
        boost_level=boost_level,
        boost_between=boost_between,
    )
    return differential_evolution(
        func,
        bounds=[
            psm_fdr, pep_fdr, prot_fdr, link_fdr, ppi_fdr
        ],
        strategy='best2bin',
        disp=True,
        mutation=(0.01, 0.5),
        popsize=15,
        workers=3
    )


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
