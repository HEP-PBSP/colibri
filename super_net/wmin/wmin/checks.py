"""
TODO
"""
from reportengine.checks import make_check, CheckError

@make_check
def check_wminpdfset_is_montecarlo(ns, **kwargs):
    """
    same as `validphys.checks.check_pdf_is_montecarlo` but for
    weight minimization set
    """
    
    pdf = ns['wminpdfset']
    etype = pdf.error_type
    if etype != 'replicas':
        raise CheckError(f"Error type of PDF {pdf} must be 'replicas' and not {etype}")