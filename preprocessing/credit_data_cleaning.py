"""
Clean Creditreform Dataset.

Following steps are undertaken:
    - Select only observations between 1997-2002.
    - Select only observations from the main five industries.
    - remove observations with 0 values in variables used for denominator.
"""

import pandas as pd
import os
import collections

data_path = os.path.join('data', 'credit.csv')
named_cols = ['ID', 'T2', 'JAHR']
var_cols = ['VAR' + str(i) for i in range(1, 29)]

credit = pd.read_csv(data_path,
                     sep=';',
                     usecols=named_cols + var_cols,
                     low_memory=False)

credit[['VAR26']] = credit[['VAR26']].astype(str)

# Filter by year (i.e. take observations from 1996 to 2002).
credit = credit[(credit['JAHR'] > 1996) & (credit['JAHR'] <= 2002)]

# Filter by asset size (VAR6).
credit = credit[(credit['VAR6'] >= 10 ** 5) & (credit['VAR6'] <= 10 ** 8)]

# Filter out companies with 0 values in variables used in the denominator.
credit = credit[
    (credit['VAR6'] != 0) &
    (credit['VAR16'] != 0) &
    (credit['VAR1'] != 0) &
    (credit['VAR2'] != 0) &
    (credit['VAR12'] != 0) &
    (credit['VAR12']+credit['VAR13'] != 0) &
    (credit['VAR6']-credit['VAR5']-credit['VAR1']-credit['VAR8'] != 0) &
    (credit['VAR19'] != 0)]


# Generate a defaultdict ind, which defaults to other.
ind = collections.defaultdict(lambda: 'other')


# Update defaultdict with the 4 industry categories.
def add_category(cat, iterable):
    global ind
    for id in iterable:
        ind[str(id)] = cat


categories = ['manufacturing', 'wholesale', 'construction', 'real_estate']

add_category('real_estate', range(70, 75))
add_category('wholesale', range(50, 53))
add_category('construction', range(45, 46))
add_category('manufacturing', range(15, 37))

# Match category id with industry class from WZ 93 by 2 digits and store it.
credit['category'] = credit[['VAR26']].apply(lambda l: [ind[s[:2]] for s in l])

# Remove 'other' category.
credit = credit[credit['category'] != 'other']

# Print summary
ind_cat = credit.groupby(['T2']).category.value_counts(normalize=True)
print(ind_cat)
print(credit.groupby(['T2'])['ID'].nunique())


# Replace category by dummies.
dummies_category = pd.get_dummies(credit.category)
dummies_VAR27 = pd.get_dummies(credit.VAR27)
credit = pd.concat([credit, dummies_category, dummies_VAR27], axis=1)

# Write to csv.
data_out = os.path.join('data', 'credit_clean.csv')
cols = [c for c in credit.columns if c not in ['VAR26', 'VAR27', 'category']]
credit.to_csv(data_out, sep=';', columns=cols, index=False)
