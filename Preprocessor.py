import pandas


def process_file(filepath, data_columns):
    file = pandas.read_csv(filepath)
    processed_file = file[data_columns]
    processed_file.to_csv(f'processed.csv', index=False)
