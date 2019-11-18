import pandas


class Preprocessor:

    def __init__(self):
        self.data_columns = ['ListYear', 'Town', 'Address', 'AssessedValue', 'PropertyType', 'SaleAmount']

    def process_file(self, filepath):
        file = pandas.read_csv(filepath)
        processed_file = file[self.data_columns]
        processed_file.to_csv(f'processed.csv', index=False)


ps = Preprocessor()
ps.process_file('data.csv')
