# pipelines/behavioral/nodes.py  (thin wrapper)
from credit_scoring.behavioral.build import build_behavioral_all_months

def build_behavioral_node(production, transactions, params):
    return build_behavioral_all_months(transactions, production, params)