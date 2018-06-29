import sys,getopt
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import math
import pickle

model_file = 'model.sav'
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
    # model = linear_reg(X_train,Y_train)
    model = gd_reg(X_train, Y_train)
    model_stats(model,X_validation,Y_validation)
    pickle.dump(model, open(model_file, 'wb'))
    print("model saved")
    return model


def linear_reg(X_train,Y_train):
    lm = LinearRegression()
    lm.fit(X_train, Y_train)
    print("done training using linear regressor...")
    return lm


def random_forest(X_train,Y_train):
    forest_reg = RandomForestRegressor(random_state=42)
    forest_reg.fit(X_train, Y_train)
    print("done training using random forest regressor...")
    return forest_reg


def gd_reg(X_train,Y_train):
    gd = ensemble.GradientBoostingRegressor()
    gd.fit(X_train, Y_train)
    print("done training using gradient boost regressor...")
    return gd


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
    disputed = {'Yes': 1, 'No': 0}
    dataset['disputed'] = dataset['Disputed'].map(disputed)

    if cust_avg_settled is None:
        cust_avg_df = pd.read_csv('avg_days.csv')
        cust_avg_settled = pd.Series(cust_avg_df.avgDaysToSettle.values, index=cust_avg_df.customerID).to_dict()

    dataset['avgDaysToSettle'] = dataset['customerID'].map(cust_avg_settled)
    if cust_dict is None:
        cust_map_df = pd.read_csv('customer_map.csv')
        cust_dict = pd.Series(cust_map_df.code.values, index=cust_map_df.customerID).to_dict()

    dataset['cust'] = dataset['customerID'].map(cust_dict)
    dataset_final = dataset[['cust', 'InvoiceAmount', 'invoicemonth', 'monthend', 'firsthalfmonth', 'paperless', 'disputed', 'avgDaysToSettle','DaysToSettle']]
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

    loaded_model = pickle.load(open(model_file, 'rb'))
    y_value = loaded_model.predict(x_value)
    # print("prediction: ")
    # print(y_value)
    invoice_data['predicted'] = y_value
    # print(invoice_data)
    get_predicted_settled_date(invoice_data)
    build_graphs(invoice_data)


def get_predicted_settled_date(invoice_data):

    invoice_data['predictedDate'] = pd.to_datetime(invoice_data.InvoiceDate) + pd.to_timedelta(pd.np.ceil(invoice_data.predicted),unit="D")
    invoice_data['predictedDate'] = invoice_data['predictedDate'].dt.strftime('%m/%d/%Y')
    out = invoice_data[['countryCode','customerID','invoiceNumber','InvoiceDate','DueDate','InvoiceAmount','Disputed','PaperlessBill','predictedDate']].copy()
    print(out)


def build_graphs(invoice_data):
    invoice_cash=invoice_data[['predictedDate','InvoiceAmount']].copy()
    invoice_cash = invoice_cash.sort_values(by='predictedDate')
    invoice_cash = invoice_cash.assign(sum=invoice_cash.InvoiceAmount.cumsum())
    invoice_cash['sum']=invoice_cash['sum'].round()
    # print(invoice_cash.head(2))
    invoice_bar=invoice_data[['predicted','invoiceNumber']].copy()
    invoice_bar['ontime'] = np.where(invoice_bar['predicted']<30,1,0)
    invoice_bar['delayed10'] = np.where(((invoice_bar['predicted']>30) & (invoice_bar['predicted']<40)),1,0)
    invoice_bar['delayed30'] = np.where(((invoice_bar['predicted']>40) & (invoice_bar['predicted']<60)),1,0)
    invoice_bar['delayed30p'] = np.where((invoice_bar['predicted']>60),1,0)
    ontime=invoice_bar['ontime'].sum()
    delayed10=invoice_bar['delayed10'].sum()
    delayed30=invoice_bar['delayed30'].sum()
    delayed30p=invoice_bar['delayed30p'].sum()
    array = invoice_cash.values
    x = array[:, 0:1]
    y = array[:, 2:3]
    n = len(x)
    x = np.reshape(x, n)
    y = np.reshape(y, n)
    # write as json
    var = '{"label1": ["' + '","'.join(x) + '"' + '] ,"data1": [' + ','.join(map(str, y)) + '],'+'"data2":['+str(ontime)+','+str(delayed10)+','+str(delayed30)+','+str(delayed30p)+']}'
    # print(var)
    text_file = open("out.json", "w")
    text_file.write(var)
    text_file.close()
    # print(invoice_bar.head(2))


if __name__ == "__main__":
   main(sys.argv[1:])