import Preprocessor as prepro
import Forests as frst
import DecisionTrees as dt
import Regression as rgr
import FileCleaner


def run_main():
    # self.data_columns = ['ListYear', 'Town', 'Address', 'AssessedValue', 'PropertyType', 'SaleAmount']
    data_columns = ['ListYear', 'Town', 'Address', 'PropertyType', 'AssessedValue', 'SaleAmount']

    print('Reformat file? (y/n)')
    user_input = input()
    if user_input is 'y' or user_input is 'Y':
        prepro.process_file('data.csv', data_columns)

    print('Clean file? (y/n)')
    user_input = input()

    if user_input is 'y' or user_input is 'Y':
        FileCleaner.clean('processed.csv')

    keep_running = True

    while keep_running:
        print()
        print('Which algorithm do you want to run? (Enter number)')
        print('1. Regression')
        print('2. Decision Tree')
        print('3. Random Forest')
        print('4. (GridSearch) Random Forest')
        print('5. (Exit Program)')
        user_input = input()

        if user_input is '1':
            print('----------------------------------------------------------------')
            rgr.run_kfold()
            print('----------------------------------------------------------------')
        elif user_input is '2':
            print('----------------------------------------------------------------')
            dt.run_gsearch()
            print('----------------------------------------------------------------')
        elif user_input is '3':
            print('----------------------------------------------------------------')
            frst.run()
            print('----------------------------------------------------------------')
        elif user_input is '4':
            print('----------------------------------------------------------------')
            frst.run_grid_search()
            print('----------------------------------------------------------------')
        elif user_input is '5':
            keep_running = False
            print('Goodbye!')
            exit(0)
        else:
            print("Please enter a valid number.")
            print("Example: To select Regression, type the number 1 and press enter.")


run_main()
