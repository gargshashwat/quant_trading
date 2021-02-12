import mysql.connector as mysql
import pandas as pd
import numpy as np
from datetime import date
import yfinance as yf
from tqdm import tqdm as tqdm

yf.pdr_override()


def get_russell_3000_tickers():
    """get tickers for Russell 3000 from a locally saved file"""
    tickers = pd.read_excel('Russell3000.xls').Ticker.tolist()
    return [str(t).upper() for t in tickers]  # ticker TRUE otherwise interpreted as bool


def create_price_volume_table(tickers, start_date, end_date):
    """store price volume data in database for given tickers and date range"""
    # create database
    db = mysql.connect(host='localhost', user='root', passwd='evanescence-1')
    cursor = db.cursor(buffered=True)
    query = """CREATE DATABASE IF NOT EXISTS stocks"""
    cursor.execute(query)

    db = mysql.connect(host='localhost', user='root', passwd='evanescence-1', database='stocks')
    cursor = db.cursor(buffered=True)

    # create table
    query = """CREATE TABLE IF NOT EXISTS price_volume_US(
            ticker VARCHAR(5),
            date DATE,
            open FLOAT,
            high FLOAT,
            low FLOAT,
            close FLOAT,
            volume FLOAT,
            dividends FLOAT,
            stock_splits FLOAT,
            PRIMARY KEY (ticker, date)
            )"""
    cursor.execute(query)

    # load stocks data and insert into table
    for t in tqdm(tickers):
        df = yf.Ticker(t).history(start=start_date, end=end_date)
        if df.empty:
            continue
        df = df.fillna(0)

        query = """INSERT IGNORE INTO price_volume_US (ticker,date,open,high,low,close,volume,dividends,stock_splits)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
        values = [(t, row.name.date(), float(row.Open), float(row.High), float(row.Low), float(row.Close),
                   float(row.Volume), float(row.Dividends), float(row['Stock Splits'])) for _,row in df.iterrows()]
        cursor.executemany(query, values)
        db.commit()


def create_shares_outstanding_table(tickers):
    """store shares outstanding data in database for given tickers and date = today"""
    # create database
    db = mysql.connect(host='localhost', user='root', passwd='evanescence-1')
    cursor = db.cursor(buffered=True)
    query = """CREATE DATABASE IF NOT EXISTS stocks"""
    cursor.execute(query)

    db = mysql.connect(host='localhost', user='root', passwd='evanescence-1', database='stocks')
    cursor = db.cursor(buffered=True)

    # create table
    query = """CREATE TABLE IF NOT EXISTS shares_outstanding_US(
            ticker VARCHAR(5),
            date DATE,
            shares_outstanding FLOAT,
            PRIMARY KEY (ticker, date)
            )"""
    cursor.execute(query)

    # load stocks data and insert into table
    values = []
    for t in tqdm(tickers):
        try:
            out = yf.Ticker(t).get_info()['sharesOutstanding']
        except:
            try:
                out = yf.Ticker(t).get_info()
                out = out['marketCap'] // out['previousClose']
            except:
                print(f'No data found for {t}')
                continue
        if np.isnan(out):
            out = 0
        values.append((t, date.today(), out))

    query = """INSERT IGNORE INTO shares_outstanding_US (ticker,date,shares_outstanding)
               VALUES (%s,%s,%s)"""
    cursor.executemany(query, values)
    db.commit()


def store_mapping(tickers):
    """store a mapping between tickers and an integer id.
    This is not to be changed at any time. New entries can only be appended."""
    # create database
    db = mysql.connect(host='localhost', user='root', passwd='evanescence-1')
    cursor = db.cursor(buffered=True)
    query = """CREATE DATABASE IF NOT EXISTS stocks"""
    cursor.execute(query)

    db = mysql.connect(host='localhost', user='root', passwd='evanescence-1', database='stocks')
    cursor = db.cursor(buffered=True)

    # create table
    query = """CREATE TABLE IF NOT EXISTS mapping(
                ticker VARCHAR(5),
                id INT,
                PRIMARY KEY (id)
                )"""
    cursor.execute(query)

    # find max id in table so far
    query = """SELECT MAX(id) FROM mapping"""
    cursor.execute(query)
    if cursor.fetchall()[0] == (None,):
        max_id = 0
    else:
        max_id = cursor.fetchall()[0][0]

    # find tickers already inserted
    query = "SELECT ticker FROM mapping"
    cursor.execute(query)
    list_ids = [x[0] for x in cursor.fetchall()]

    for t in tqdm(tickers):
        if t not in list_ids:
            query = """INSERT IGNORE INTO mapping (ticker,id)
                           VALUES (%s,%s)"""
            values = (t, max_id + 1)
            cursor.execute(query, values)
            db.commit()
            max_id += 1


def create_gspc_table(start_date, end_date):
    """store price volume data in database for S&P500"""
    # create database
    db = mysql.connect(host='localhost', user='root', passwd='evanescence-1')
    cursor = db.cursor(buffered=True)
    query = """CREATE DATABASE IF NOT EXISTS stocks"""
    cursor.execute(query)

    db = mysql.connect(host='localhost', user='root', passwd='evanescence-1', database='stocks')
    cursor = db.cursor(buffered=True)

    # create table
    query = """CREATE TABLE IF NOT EXISTS sp500_US(
            date DATE,
            close FLOAT,
            PRIMARY KEY (date)
            )"""
    cursor.execute(query)

    # load stocks data and insert into table
    df = yf.Ticker('^GSPC').history(start=start_date, end=end_date)[['Close']]
    if df.empty:
        return
    df = df.fillna(0)

    query = """INSERT IGNORE INTO sp500_US (date,close)
                VALUES (%s,%s)"""
    values = [(row.name.date(), float(row.Close)) for _,row in df.iterrows()]
    cursor.executemany(query, values)
    db.commit()


def create_vix_table(start_date, end_date):
    """store price volume data in database for VIX"""
    # create database
    db = mysql.connect(host='localhost', user='root', passwd='evanescence-1')
    cursor = db.cursor(buffered=True)
    query = """CREATE DATABASE IF NOT EXISTS stocks"""
    cursor.execute(query)

    db = mysql.connect(host='localhost', user='root', passwd='evanescence-1', database='stocks')
    cursor = db.cursor(buffered=True)

    # create table
    query = """CREATE TABLE IF NOT EXISTS vix_US(
            date DATE,
            close FLOAT,
            PRIMARY KEY (date)
            )"""
    cursor.execute(query)

    # load stocks data and insert into table
    df = yf.Ticker('^VIX').history(start=start_date, end=end_date)[['Close']]
    if df.empty:
        return
    df = df.fillna(0)

    query = """INSERT IGNORE INTO vix_US (date,close)
                VALUES (%s,%s)"""
    values = [(row.name.date(), float(row.Close)) for _,row in df.iterrows()]
    cursor.executemany(query, values)
    db.commit()


# tickers = get_russell_3000_tickers()
# store_mapping(tickers)
#
# db = mysql.connect(host='localhost', user='root', passwd='evanescence-1', database='stocks')
# cursor = db.cursor(buffered=True)
#
# query = """SELECT * FROM mapping"""
# cursor.execute(query)
# for x in cursor.fetchall():
#     print(x)
