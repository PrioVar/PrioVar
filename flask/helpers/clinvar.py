import pandas
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from os import path
from config import uri, username, password

def read_clinvar():
    df = pd.read_csv(path.join("data", "clinvar_20231121.vcf"), sep='\t', comment='#', header=None)
    df.columns = ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO']
    return df

def save_clinvar(df: pandas.DataFrame):
    driver = GraphDatabase.driver(uri, auth=(username, password))
    with driver.session() as session:
        for index, row in df.iterrows():
            session.run("CREATE (a:Variant {id: $id, chrom: $chrom, pos: $pos, ref: $ref, alt: $alt, qual: $qual, filter: $filter, info: $info, isClinVar: $isClinVar})",
                        id=row['ID'], chrom=row['CHROM'], pos=row['POS'], ref=row['REF'], alt=row['ALT'], qual=row['QUAL'], filter=row['FILTER'], info=row['INFO'], isClinVar=True)