import csv
from utils.models import DataInstance


def read_csv_file(csv_file_path, delimiter='\t'):
    """
     This function takes file path as an argument, reads the data file and
     returns a list of DataInstance objects with text and true labels

    :param delimiter: Delimiter for the file. Default is Tab(\t)
    :param csv_file_path: path to the TSV/CSV file
    :return: returns a list of  DataInstance class objects. <utils.models.DataInstance>
    """
    with open(csv_file_path, 'r') as file:
        file_data = csv.reader(file, delimiter=delimiter)
        data = []
        for row in file_data:
            data.append(DataInstance(row[0], row[2], row[3]))
        return data
