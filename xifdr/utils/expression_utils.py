import polars as pl
import json
import io


def replace_input(expr, col_name):
    expr_json = json.loads(
        expr.meta.serialize(format='json')
    )
    return pl.Expr.deserialize(
        io.StringIO(json.dumps(_replace_input_json(expr_json, col_name))),
        format='json'
    )

def _replace_input_json(expr_json, col_name):
    if isinstance(expr_json, dict):
        for k, v in expr_json.items():
            if k == 'Column':
                expr_json[k] = col_name
            else:
                _replace_input_json(expr_json[k], col_name)
    if isinstance(expr_json, list):
        for e in expr_json:
            _replace_input_json(e, col_name)
    return expr_json