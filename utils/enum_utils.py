def find_enum_by_value(enum_class, value):
    values_fit = [c for c in enum_class if c.value == value]
    if len(values_fit) < 1:
        raise ValueError(f'Member of {enum_class} not found for value: {value}')
    if len(values_fit) > 1:
        raise ValueError(f'Found multiple values of {enum_class} for value {value}: {values_fit}')
    return values_fit[0]
