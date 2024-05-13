from neo4j import GraphDatabase
from config import uri, username, password
import base64
import pandas as pd
import tempfile
import os


def read_file_content_and_return_df(vcf_id):
    driver = GraphDatabase.driver(uri, auth=(username, password))

    # get the VCFFile with the given vcf_id, use ID() to access
    with driver.session() as session:
        result = session.run(
            "MATCH (v:VCFFile) WHERE ID(v) = $vcf_id RETURN v", {"vcf_id": vcf_id}
        )
        vcf_file = result.single()[0]

    if vcf_file and 'content' in vcf_file:
        base64_content = vcf_file['content']
        decoded_content = base64.b64decode(base64_content)

        # Write to a temporary file
        temp = tempfile.NamedTemporaryFile(delete=False)
        try:
            temp.write(decoded_content)
            temp.close()

            # Read VCF into DataFrame
            df = pd.read_csv(temp.name, comment='#', delimiter='\t', header=None,
                             names=["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", "SAMPLE"],
                             dtype={"CHROM": str, "POS": int, "ID": str, "REF": str, "ALT": str, "QUAL": str,
                                    "FILTER": str, "INFO": str, "FORMAT": str, "SAMPLE": str})
            return df
        except Exception as e:
            return str(e)
        finally:
            os.remove(temp.name)
    else:
        return "No file content found"


def read_file_content(vcf_id):
    driver = GraphDatabase.driver(uri, auth=(username, password))

    # get the VCFFile with the given vcf_id, use ID() to access
    with driver.session() as session:
        result = session.run(
            "MATCH (v:VCFFile) WHERE ID(v) = $vcf_id RETURN v", {"vcf_id": vcf_id}
        )
        vcf_file = result.single()[0]

    if vcf_file and 'content' in vcf_file:
        base64_content = vcf_file['content']
        decoded_content = base64.b64decode(base64_content)
        return decoded_content
    else:
        return "No file content found"
