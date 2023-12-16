import pandas as pd

def annotate_variants(file):
    # read the file
    df = pd.read_csv(file, sep='\t', comment='#', header=None)
    print(df.columns)
    return df