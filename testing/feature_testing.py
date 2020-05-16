import os
from utils.csv import read_csv_file

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train_file_path = project_root+'/data/tsv/train.tsv'
test_file_path = project_root+'/data/tsv/test.tsv'

print(train_file_path)

data = read_csv_file(csv_file_path=train_file_path, delimiter='\t')

i = 0
feature_dict = {}
for inst in data:
    if len(inst.features) >= 0:
        # inst.print()
        i += 1
        tokens = inst.text.split()
        for token in tokens:
            if token not in feature_dict:
                feature_dict[token] = 1
                continue
            feature_dict[token] += 1

for key in sorted(feature_dict, key=feature_dict.get, reverse=True):
    print(key, ' -> ', feature_dict.get(key))
# print('Data Points without Features :: ', i)

#         tokens = inst.text.split()
#         for token in tokens:
#             if token not in feature_dict:
#                 feature_dict[token] = 1
#                 continue
#             feature_dict[token] += 1
#
# for key in sorted(feature_dict, key=feature_dict.get, reverse=True):
#     print(key, ' -> ', feature_dict.get(key))
