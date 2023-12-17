import pandas as pd
import requests
import copy

def annotate_variants(file):
    # send an HTTP Request to the annotation service
    # http://127.0.0.1:{PORT}/api/v1/vcf/ where PORT is 9096
    # the request should contain the file and authorization token as header
    token = "a66c6eebcf2c4d106abc0447a85b7f72de570477"
    port = 9096
    response = requests.post('http://127.0.0.1:{PORT}/api/v1/vcf/'.format(PORT=port), files={'file': file},
                              headers={"Authorization": "Token {TOKEN}".format(TOKEN=token),
                                       "Content-Disposition": f"attachment; filename={file.name}"})

    # print the contents of the response
    print(response.content)
    print(response.status_code)

    response = requests.post('http://127.0.0.1:{PORT}/api/v1/vcf/job'.format(PORT=port), files={'file': file},
                              headers={"Authorization": "Token {TOKEN}".format(TOKEN=token),
                                       "Content-Disposition": f"attachment; filename={file.name}"})

    # read the file
    df = pd.read_csv(file, sep='\t', comment='#', header=None)
    print(df.columns)
    return df