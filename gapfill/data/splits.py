import numpy as np
def loso_sites(all_sites, val_site):
    return [s for s in all_sites if s != val_site], [val_site]
def loyo_years(all_years, val_year):
    train = [y for y in all_years if y != val_year]
    return train, [val_year]
