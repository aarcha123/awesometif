import sys,getopt
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
import math
import pickle

lr_model_file = 'lr_model.sav'
model = None
X_test = None
Y_test = None

cust_dict = None
cust_avg_settled = None


def main(argv):
    trainfile=''
    testfile=''
    datafile=''
    try:
        opts, args = getopt.getopt(argv, "ht:e:p", ["train=", "test=","predict="])
    except getopt.GetoptError:
        print('test.py -t <traindata> -v <validatedata> -p <predictdata>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -t <traindata> -v <validatedata> -p <predictdata>')
            sys.exit()
        elif opt in ("-t", "--train"):
            train(arg)
        elif opt in ("-e", "--test"):
            testfile = arg
        elif opt in ("-p", "--predict"):
            predict(arg)


def train(trainfile):
    global model
    dataset = pd.read_csv(trainfile)
    dataset_new = extract_features(dataset)
    array = dataset_new.values
    n = len(dataset_new.columns)
    X = array[:, 0:n - 1]
    Y = array[:, n - 1]
    seed = 7
    X_train, X_rest, Y_train, Y_rest = model_selection.train_test_split(X, Y, test_size=0.40, random_state=seed)
    X_validation, X_test, Y_validation, Y_test = model_selection.train_test_split(X_rest, Y_rest, test_size=0.50,random_state=seed)
    lm = LinearRegression()
    model = lm
    lm.fit(X_train, Y_train)
    print("done training...")
    model_stats(lm,X_validation,Y_validation)
    pickle.dump(lm, open(lr_model_file, 'wb'))
    print("model saved")
    return lm


def model_stats(lm,X_validation, Y_validation):
    print("score= ", lm.score(X_validation, Y_validation))
    y_predict = lm.predict(X_validation)
    regression_model_mse = mean_squared_error(y_predict, Y_validation)
    print("regression rmse:", math.sqrt(regression_model_mse))


def extract_features(dataset):
    global cust_avg_settled
    global cust_dict
    grouped = dataset.groupby('customerID', as_index=False)
    invoice_count = grouped.agg({"invoiceNumber": "count"})
    invoice_count.columns = ['customerID', 'total']

    custlist = invoice_count['customerID'].tolist()
    cust_dict = {x: custlist.index(x) for x in custlist}

    df = pd.DataFrame(list(cust_dict.items()), columns=['customerID', 'code'])

    df.to_csv("customer_map.csv", index=0)

    settled_days_avg = grouped.agg({'DaysToSettle': 'mean'})
    settled_days_avg.columns = ['customerID', 'avgDaysToSettle']

    settled_days_avg.to_csv("avg_days.csv", index=0)
    cust_avg_settled = pd.Series(settled_days_avg.avgDaysToSettle.values, index=settled_days_avg.customerID).to_dict()
    dataset_enriched = calc_features(dataset)
    return dataset_enriched


def calc_features(dataset):
    global cust_avg_settled
    global cust_dict
    dataset['invoicemonth'] = pd.to_datetime(dataset['InvoiceDate']).dt.month
    dataset['invoicedate'] = pd.to_datetime(dataset['InvoiceDate']).dt.day
    dataset['invoiceday'] = pd.to_datetime(dataset['InvoiceDate']).dt.weekday
    dataset['monthend'] = np.where(dataset['invoicedate'] > 27, 1, 0)
    dataset['firsthalfmonth'] = np.where(dataset['invoicedate'] < 16, 1, 0)
    paperless = {'Paper': 0, 'Electronic': 1}
    dataset['paperless'] = dataset['PaperlessBill'].map(paperless)
    if cust_avg_settled is None:
        cust_avg_df = pd.read_csv('avg_days.csv')
        cust_avg_settled = pd.Series(cust_avg_df.avgDaysToSettle.values, index=cust_avg_df.customerID).to_dict()

    dataset['avgDaysToSettle'] = dataset['customerID'].map(cust_avg_settled)
    if cust_dict is None:
        cust_map_df = pd.read_csv('customer_map.csv')
        cust_dict = pd.Series(cust_map_df.code.values, index=cust_map_df.customerID).to_dict()

    dataset['cust'] = dataset['customerID'].map(cust_dict)
    dataset_final = dataset[['cust', 'InvoiceAmount', 'invoicemonth', 'monthend', 'firsthalfmonth', 'paperless', 'avgDaysToSettle','DaysToSettle']]
    cols = dataset_final.columns
    dataset_final[cols] = dataset_final[cols].apply(pd.to_numeric)
    return dataset_final


def auto_extract_feature(X_train,Y_train):
    rfe = RFE(model, 4)
    fit = rfe.fit(X_train, Y_train)
    print("Num Features: %d" % fit.n_features_)
    print("Selected Features: %s" % fit.support_)
    print("Feature Ranking: %s" % fit.ranking_)


def file_to_array(filename):
    invoice_data = pd.read_csv(filename)
    invoice_data_enriched = calc_features(invoice_data)
    array = invoice_data_enriched.values
    n = len(invoice_data_enriched.columns)
    X = array[:, 0:n - 1]
    return X


def predict(datafile):
    invoice_data = pd.read_csv(datafile)
    invoice_data_enriched = calc_features(invoice_data)
    array = invoice_data_enriched.values
    n = len(invoice_data_enriched.columns)
    x_value = array[:, 0:n - 1]

    loaded_model = pickle.load(open(lr_model_file, 'rb'))
    y_value = loaded_model.predict(x_value)
    print("prediction: ")
    print(y_value)
    invoice_data['predicted'] = y_value
    print(invoice_data.head(1))


def to_json():
    print('json')


if __name__ == "__main__":
   main(sys.argv[1:])