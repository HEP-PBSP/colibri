"""
TODO
"""

from validphys.sumrules import sum_rules
from reportengine import collect


def sum_rules_dict(pdf, Q):
    """
    TODO
    """
    return {str(pdf): sum_rules(pdf, Q)}

pdfs_sum_rules = collect("sum_rules_dict", ("pdfs",))

def basis_replica_selector(pdfs_sum_rules, sum_rule_atol=1e-2):
    """

    """