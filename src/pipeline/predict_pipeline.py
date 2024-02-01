import sys,os
from src.exception import CustomException
from src.components.prepare_data import prepare_data
import yfinance as yf
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features_df):
        try:
            model_path = os.path.join('artifacts','model.h5')
            model = tf.keras.models.load_model(model_path)
            bdf, X_test = prepare_data(features_df)
            pred = model.predict(X_test)
            bdf['predict'] = pred
            
            return bdf
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,
        HK_stock_symbol:str,
        US_stock_symbol:str,
        Crypto_symbol:str,
        Symbol_type:str,
        Enter_symbol:str
        ):

        self.HK_stock_symbol = HK_stock_symbol
        self.US_stock_symbol = US_stock_symbol
        self.Crypto_symbol = Crypto_symbol
        self.Symbol_type = Symbol_type
        self.Enter_symbol = Enter_symbol



    def get_data_as_data_frame(self):
        try:
            df = yf.download(f'{self.HK_stock_symbol}.HK',start='2022-10-1')
            df.reset_index(inplace=True)
            df.columns = df.columns.str.lower()
            HK_stock_df = df[['date','open','high','low','close','volume']]

            df = yf.download(f'{self.US_stock_symbol}',start='2022-10-1')
            df.reset_index(inplace=True)
            df.columns = df.columns.str.lower()
            US_stock_df = df[['date','open','high','low','close','volume']]

            df = yf.download(f'{self.Crypto_symbol}-USD',start='2022-10-1')
            df.reset_index(inplace=True)
            df.columns = df.columns.str.lower()
            Crypto_df = df[['date','open','high','low','close','volume']]

            if self.Symbol_type == 'HK_stock':
                df = yf.download(f'{self.Enter_symbol}.HK',start='2022-10-1')
            elif self.Symbol_type == 'US_stock':
                df = yf.download(f'{self.Enter_symbol}',start='2022-10-1')
            elif self.Symbol_type == 'Crypto':
                df = yf.download(f'{self.Enter_symbol}-USD',start='2022-10-1')
            df.reset_index(inplace=True)
            df.columns = df.columns.str.lower()
            Enter_df = df[['date','open','high','low','close','volume']]

            return {'HK_stock_df':HK_stock_df, 'US_stock_df':US_stock_df, 'Crypto_df':Crypto_df, 'Enter_df':Enter_df}
        except Exception as e:
            raise CustomException(e, sys)