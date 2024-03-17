from tqdm import tqdm

def parse_tsv_file(input_file_path, output_file_path):
    header = "#CHROM\tPOS\tREF\tALT\tAM_PATHOGENICITY\tPATHOGENICITY_CLASS\n"
    with open(input_file_path, 'r') as input_file:
        with open(output_file_path, 'w') as vcf_file:
            vcf_file.write(header)
            for _ in range(4):  # Skip the first 4 lines
                next(input_file)
            for line in tqdm(input_file):
                columns = line.strip().split('\t')
                # Select desired columns, e.g., columns 0, 1, 2, 3, 8, 9(CHROM, POS, REF, ALT, AM_PATHOGENICITY, PATHOGENICITY_CLASS)
                selected_columns = (columns[0], columns[1], columns[2], columns[3], columns[8], columns[9])
                # Write selected columns to the VCF file
                vcf_file.write('\t'.join(selected_columns) + '\n')



# Example usage:
input_file_path = 'AlphaMissense_hg38.tsv'
output_file_path = 'AlphaMissense_variants_pathogenicity.vcf'
print("Starting to parse...")
parsed_data = parse_tsv_file(input_file_path, output_file_path)
print( "Parsing completed!")