def _orca_factory():
    from policy.orca import ORCA
    return ORCA()

policy_factory = dict()
policy_factory['orca'] = _orca_factory
