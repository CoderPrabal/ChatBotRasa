import pandas as pd
df = pd.read_csv('adult.tar.gz', compression='gzip', header=0, sep=' ', quotechar='"', error_bad_lines=False)
print(pd)