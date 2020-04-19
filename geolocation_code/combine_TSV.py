import os
import glob
import pandas as pd
import csv
import build_data as d
#os.chdir("/mydir")

# def set_folder(gran):
#     info_path = "data/" + gran+"/infos/"
#     os.chdir(os.path.abspath(info_path))


# set_folder('cities')
# extension = 'tsv'
# all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
# combined_csv = pd.concat([pd.read_csv(f,sep='\t',) for f in all_filenames ])

#combined_csv.to_csv("dataset_cities_infos.tsv", index=False, encoding='utf-8-sig', sep='\t')
info_path = "data/" + 'cities/'
reader=csv.reader(open(info_path+"dataset_infos.tsv"),delimiter='\t')
row_count = sum(1 for row in reader)
print (row_count)

# for file in d._get_files(info_path):
#     print (file)

#     reader=csv.reader(open(file),delimiter='\t')
#     row_count += sum(1 for row in reader)
# print (row_count)
#12300093
#12331767
#12331862
#12331893