import re
import warnings

def guess_column_names(columns:list[str], static_mapping:dict = None):
    name_mapping = static_mapping.copy() if static_mapping else {}
    if 'score' not in name_mapping.values():
        name_mapping[_guess_score(columns)] = 'score'
    if 'sequence_p1' not in name_mapping.values():
        name_mapping[_guess_sequence(columns)[0]] = 'sequence_p1'
    if 'sequence_p2' not in name_mapping.values():
        name_mapping[_guess_sequence(columns)[1]] = 'sequence_p2'
    if 'start_pos_p1' not in name_mapping.values():
        name_mapping[_guess_start_pos(columns)[0]] = 'start_pos_p1'
    if 'start_pos_p2' not in name_mapping.values():
        name_mapping[_guess_start_pos(columns)[1]] = 'start_pos_p2'
    if 'link_pos_p1' not in name_mapping.values():
        name_mapping[_guess_link_pos(columns)[0]] = 'link_pos_p1'
    if 'link_pos_p2' not in name_mapping.values():
        name_mapping[_guess_link_pos(columns)[1]] = 'link_pos_p2'
    if 'charge' not in name_mapping.values():
        name_mapping[_guess_charge(columns)] = 'charge'
    if 'protein_p1' not in name_mapping.values():
        name_mapping[_guess_protein(columns)[0]] = 'protein_p1'
    if 'protein_p2' not in name_mapping.values():
        name_mapping[_guess_protein(columns)[1]] = 'protein_p2'
    if 'decoy_p1' not in name_mapping.values():
        name_mapping[_guess_decoy(columns)[0]] = 'decoy_p1'
    if 'decoy_p2' not in name_mapping.values():
        name_mapping[_guess_decoy(columns)[1]] = 'decoy_p2'
    if 'fdr_group' not in name_mapping.values():
        name_mapping[_guess_fdr_group(columns)] = 'fdr_group'
    if 'coverage_p1' not in name_mapping.values() or 'coverage_p2' not in name_mapping.values():
        coverage_guess = _guess_coverage(columns)
        if coverage_guess:
            if 'coverage_p1' not in name_mapping.values():
                name_mapping[coverage_guess[0]] = 'coverage_p1'
            if 'coverage_p2' not in name_mapping.values():
                name_mapping[coverage_guess[1]] = 'coverage_p2'
    return name_mapping

def _guess_score(columns: list[str]):
    columns_lower = [c.lower() for c in columns]
    # Guess score column
    if 'score' in columns:
        return 'score'
    if 'score' in columns_lower:
        return _first_lower('score', columns)
    if 'match_score' in columns_lower:
        # xisearch2
        return _first_lower('match_score', columns)
    if 'classificationscore' in columns_lower:
        # scout
        return _first_lower('classificationscore', columns)
    if 'xlscore' in columns_lower:
        # scout fallback
        return _first_lower('xlscore', columns)
    for c in columns:
        if 'score' in c.lower():
            warnings.warn(f'Wild guess for score column: {c}.')
            return c
    raise NoColumnNameGuess("Could not guess score column.")

def _guess_sequence(columns: list[str]):
    columns_lower = [c.lower() for c in columns]
    if 'sequence_p1' in columns_lower and 'sequence_p2' in columns_lower:
        # xisearch2
        return (
            _first_lower('sequence_p1', columns),
            _first_lower('sequence_p2', columns),
        )
    if 'sequence1' in columns_lower and 'sequence2' in columns_lower:
        return (
            _first_lower('sequence1', columns),
            _first_lower('sequence2', columns),
        )
    if 'pepseq1' in columns_lower and 'pepseq2' in columns_lower:
        return (
            _first_lower('pepseq1', columns),
            _first_lower('pepseq2', columns),
        )
    if 'peptide1' in columns_lower and 'peptide2' in columns_lower:
        return (
            _first_lower('peptide1', columns),
            _first_lower('peptide2', columns),
        )
    if 'alphapeptide' in columns_lower and 'betapeptide' in columns_lower:
        return (
            _first_lower('AlphaPeptide', columns),
            _first_lower('BetaPeptide', columns),
        )
    raise NoColumnNameGuess(f'Could not guess sequence columns.')

def _guess_start_pos(columns: list[str]):
    columns_lower = [c.lower() for c in columns]
    if 'start_pos_p1' in columns_lower and 'start_pos_p2' in columns_lower:
        # xisearch2
        return (
            _first_lower('start_pos_p1', columns),
            _first_lower('start_pos_p2', columns),
        )
    if _re_in('pep.*pos.*1', columns_lower) and _re_in('pep.*pos.*2', columns_lower):
        return (
            _first_lower('pep.*pos.*1', columns),
            _first_lower('pep.*pos.*2', columns),
        )
    if _re_in('start.*1', columns_lower) and _re_in('start.*2', columns_lower):
        return (
            _first_lower('start.*1', columns),
            _first_lower('start.*2', columns),
        )
    if _re_in('alpha.*pos.*1', columns_lower) and _re_in('beta.*pos.*1', columns_lower):
        return (
            _first_lower('alpha.*pos.*1', columns),
            _first_lower('beta.*pos.*2', columns),
        )
    raise NoColumnNameGuess(f'Could not guess start position columns.')

def _guess_link_pos(columns: list[str]):
    columns_lower = [c.lower() for c in columns]
    if 'link_pos_p1' in columns_lower and 'link_pos_p2' in columns_lower:
        # xisearch2
        return (
            _first_lower('link_pos_p1', columns),
            _first_lower('link_pos_p2', columns),
        )
    if 'alphapos' in columns_lower and 'betapos' in columns_lower:
        # scout
        return (
            _first_lower('alphapos', columns),
            _first_lower('betapos', columns),
        )
    if _re_in('link.*1', columns_lower) and _re_in('link.*2', columns_lower):
        return (
            _first_lower('link.*1', columns),
            _first_lower('link.*2', columns),
        )
    if _re_in('alpha.*pos', columns_lower) and _re_in('beta.*pos', columns_lower):
        return (
            _first_lower('alpha.*pos', columns),
            _first_lower('beta.*pos', columns),
        )
    raise NoColumnNameGuess(f'Could not guess link position columns.')

def _guess_charge(columns: list[str]):
    columns_lower = [c.lower() for c in columns]
    if 'charge' in columns_lower:
        return _first_lower('charge', columns)
    if _re_in('precursor.charge', columns_lower):
        return _first_lower('precursor.charge', columns)
    if _re_in('charge', columns_lower):
        return _first_lower('charge', columns)
    raise NoColumnNameGuess(f'Could not guess charge column.')

def _guess_protein(columns: list[str]):
    columns_lower = [c.lower() for c in columns]
    if 'protein_p1' in columns_lower and 'protein_p2' in columns_lower:
        return (
            _first_lower('protein_p1', columns),
            _first_lower('protein_p2', columns),
        )
    if 'alphamappings' in columns_lower and 'betamappings' in columns_lower:
        return (
            _first_lower('alphamappings', columns),
            _first_lower('betamappings', columns),
        )
    if _re_in('prot.*1', columns_lower) and _re_in('prot.*2', columns_lower):
        return (
            _first_lower('prot.*1', columns),
            _first_lower('prot.*2', columns),
        )
    raise NoColumnNameGuess(f'Could not guess protein columns.')

def _guess_decoy(columns: list[str]):
    columns_lower = [c.lower() for c in columns]
    if 'decoy_p1' in columns_lower and 'decoy_p2' in columns_lower:
        return (
            _first_lower('decoy_p1', columns),
            _first_lower('decoy_p2', columns),
        )
    if 'alphadecoy' in columns_lower and 'betadecoy' in columns_lower:
        return (
            _first_lower('alphadecoy', columns),
            _first_lower('betadecoy', columns),
        )
    if _re_in('.*decoy.*1', columns_lower) and _re_in('.*decoy.*2', columns_lower):
        return (
            _first_lower('.*decoy.*1', columns),
            _first_lower('.*decoy.*2', columns),
        )
    raise NoColumnNameGuess(f'Could not guess decoy columns.')

def _guess_fdr_group(columns: list[str]):
    columns_lower = [c.lower() for c in columns]
    if 'fdr_group' in columns_lower:
        return _first_lower('fdr_group', columns)
    if _re_in('fdr.*group', columns_lower):
        return _first_lower('fdr.*group', columns)
    if _re_in('.*fdr.*group.*', columns_lower):
        return _first_lower('.*fdr.*group.*', columns)
    if _re_in('.*between.*', columns_lower):
        return _first_lower('.*between.*', columns)
    if _re_in('.*self.*', columns_lower):
        return _first_lower('.*self.*', columns)
    if _re_in('.*inter.*', columns_lower):
        return _first_lower('.*inter.*', columns)
    if _re_in('.*intra.*', columns_lower):
        return _first_lower('.*intra.*', columns)
    if _re_in('.*hetero.*', columns_lower):
        return _first_lower('.*hetero.*', columns)
    if _re_in('.*homo.*', columns_lower):
        return _first_lower('.*homo.*', columns)
    warnings.warn('Could not guess fdr_group column.')


def _guess_coverage(columns: list[str]):
    columns_lower = [c.lower() for c in columns]
    if 'coverage_p1' in columns_lower and 'coverage_p2' in columns_lower:
        return (
            _first_lower('coverage_p1', columns),
            _first_lower('coverage_p2', columns),
        )
    if 'unique_peak_conservative_coverage_p1' in columns_lower \
            and 'unique_peak_conservative_coverage_p2' in columns_lower:
        return (
            _first_lower('unique_peak_conservative_coverage_p1', columns),
            _first_lower('unique_peak_conservative_coverage_p2', columns),
        )
    if 'unique.*conservative.*coverage.*1' in columns_lower and 'unique.*conservative.*coverage.*2' in columns_lower:
        return (
            _first_lower('unique.*conservative.*coverage.*1', columns),
            _first_lower('unique.*conservative.*coverage.*2', columns),
        )
    if 'unique.*coverage.*1' in columns_lower and 'unique.*coverage.*2' in columns_lower:
        return (
            _first_lower('unique.*coverage.*1', columns),
            _first_lower('unique.*coverage.*2', columns),
        )
    if 'alpha.*coverage' in columns_lower and 'beta.*coverage' in columns_lower:
        return (
            _first_lower('alpha.*coverage', columns),
            _first_lower('beta.*coverage', columns),
        )
    if _re_in('.*coverage.*1', columns_lower) and _re_in('.*coverage.*2', columns_lower):
        return (
            _first_lower('.*coverage.*1', columns),
            _first_lower('.*coverage.*2', columns),
        )
    warnings.warn('Could not guess peak coverage columns.')
    return None

def _re_in(pattern, strings):
    return any([
        s for s in strings if re.fullmatch(pattern, s)
    ])

def _first_lower(pattern, columns):
    return [
        c for c in columns if re.fullmatch(pattern, c.lower())
    ][0]

class NoColumnNameGuess(Exception):
    """Exception raised when a column name cannot be guessed."""
    def __init__(self, message="Could not guess the column name."):
        super().__init__(message)
