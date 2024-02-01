from flask import Flask, request, render_template
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method =='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            HK_stock_symbol = request.form.get('HK_stock_symbol'),
            US_stock_symbol = request.form.get('US_stock_symbol'),
            Crypto_symbol = request.form.get('Crypto_symbol'),
            Symbol_type = request.form.get('Symbol_type'),
            Enter_symbol = request.form.get('Enter_symbol')
        )
        prep_data = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        bdf_HK = predict_pipeline.predict(prep_data['HK_stock_df'])
        bdf_US = predict_pipeline.predict(prep_data['US_stock_df'])
        bdf_Crypto = predict_pipeline.predict(prep_data['Crypto_df'])
        bdf_Enter = predict_pipeline.predict(prep_data['Enter_df'])

        # plot
        img = BytesIO()
        fig, axes = plt.subplots(4,1,figsize=(10,20))
        axes[0].set_title(data.HK_stock_symbol)
        axes[0].plot(bdf_HK.date, bdf_HK.close)
        axes[0].plot(bdf_HK.date, bdf_HK.EMA_10, label='EMA 10')
        axes[0].plot(bdf_HK.date, bdf_HK.EMA_20, label='EMA_20')
        axes[0].plot(bdf_HK.date, bdf_HK.EMA_60, label='EMA_60')
        axes[0].legend()
        mask = [True if p>0 else False for p in bdf_HK.predict]
        axes[0].scatter(bdf_HK[mask]['date'], bdf_HK[mask]['close'], s=30*bdf_HK[mask]['predict'] ,c='blue',marker='o')

        axes[1].set_title(data.US_stock_symbol)
        axes[1].plot(bdf_US.date, bdf_US.close)
        axes[1].plot(bdf_US.date, bdf_US.EMA_10, label='EMA 10')
        axes[1].plot(bdf_US.date, bdf_US.EMA_20, label='EMA_20')
        axes[1].plot(bdf_US.date, bdf_US.EMA_60, label='EMA_60')
        axes[1].legend()
        mask = [True if p>0 else False for p in bdf_US.predict]
        axes[1].scatter(bdf_US[mask]['date'], bdf_US[mask]['close'], s=30*bdf_US[mask]['predict'] ,c='blue',marker='o')

        axes[2].set_title(data.Crypto_symbol)
        axes[2].plot(bdf_Crypto.date, bdf_Crypto.close)
        axes[2].plot(bdf_Crypto.date, bdf_Crypto.EMA_10, label='EMA 10')
        axes[2].plot(bdf_Crypto.date, bdf_Crypto.EMA_20, label='EMA_20')
        axes[2].plot(bdf_Crypto.date, bdf_Crypto.EMA_60, label='EMA_60')
        axes[2].legend()
        mask = [True if p>0 else False for p in bdf_Crypto.predict]
        axes[2].scatter(bdf_Crypto[mask]['date'], bdf_Crypto[mask]['close'], s=30*bdf_Crypto[mask]['predict'] ,c='blue',marker='o')

        axes[3].set_title(data.Enter_symbol)
        axes[3].plot(bdf_Enter.date, bdf_Enter.close)
        axes[3].plot(bdf_Enter.date, bdf_Enter.EMA_10, label='EMA 10')
        axes[3].plot(bdf_Enter.date, bdf_Enter.EMA_20, label='EMA_20')
        axes[3].plot(bdf_Enter.date, bdf_Enter.EMA_60, label='EMA_60')
        axes[3].legend()
        mask = [True if p>0 else False for p in bdf_Enter.predict]
        axes[3].scatter(bdf_Enter[mask]['date'], bdf_Enter[mask]['close'], s=30*bdf_Enter[mask]['predict'] ,c='blue',marker='o')

        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        print('finish')
        return render_template('home.html', plot_url=plot_url)

    
if __name__=="__main__":
    app.run(host='0.0.0.0',port=8000)
    # data = CustomData(
    #     HK_stock_symbol = '0700',
    #     US_stock_symbol = 'AAPL',
    #     Crypto_symbol = 'BTC'
    # )
    # prep_data = data.get_data_as_data_frame()
    # predict_pipeline = PredictPipeline()
    # bdf_HK = predict_pipeline.predict(prep_data['HK_stock_df'])
    # bdf_US = predict_pipeline.predict(prep_data['US_stock_df'])
    # bdf_Crypto = predict_pipeline.predict(prep_data['Crypto_df'])
    # fig, axes = plt.subplots(3,1,figsize=(10,8))
    # axes[0].set_title(data.HK_stock_symbol)
    # axes[0].plot(bdf_HK.date, bdf_HK.close)
    # axes[0].plot(bdf_HK.date, bdf_HK.EMA_10, label='EMA 10')
    # axes[0].plot(bdf_HK.date, bdf_HK.EMA_20, label='EMA_20')
    # axes[0].plot(bdf_HK.date, bdf_HK.EMA_60, label='EMA_60')
    # axes[0].legend()
    # mask = [True if p>0 else False for p in bdf_HK.predict]
    # axes[0].scatter(bdf_HK[mask]['date'], bdf_HK[mask]['close'], s=30*bdf_HK[mask]['predict'] ,c='blue',marker='o')  
    
    # axes[1].set_title(data.US_stock_symbol)
    # axes[1].plot(bdf_US.date, bdf_US.close)
    # axes[1].plot(bdf_US.date, bdf_US.EMA_10, label='EMA 10')
    # axes[1].plot(bdf_US.date, bdf_US.EMA_20, label='EMA_20')
    # axes[1].plot(bdf_US.date, bdf_US.EMA_60, label='EMA_60')
    # axes[1].legend()
    # mask = [True if p>0 else False for p in bdf_US.predict]
    # axes[1].scatter(bdf_US[mask]['date'], bdf_US[mask]['close'], s=30*bdf_US[mask]['predict'] ,c='blue',marker='o')  
    
    # axes[2].set_title(data.Crypto_symbol)
    # axes[2].plot(bdf_Crypto.date, bdf_Crypto.close)
    # axes[2].plot(bdf_Crypto.date, bdf_Crypto.EMA_10, label='EMA 10')
    # axes[2].plot(bdf_Crypto.date, bdf_Crypto.EMA_20, label='EMA_20')
    # axes[2].plot(bdf_Crypto.date, bdf_Crypto.EMA_60, label='EMA_60')
    # axes[2].legend()
    # mask = [True if p>0 else False for p in bdf_Crypto.predict]
    # axes[2].scatter(bdf_Crypto[mask]['date'], bdf_Crypto[mask]['close'], s=30*bdf_Crypto[mask]['predict'] ,c='blue',marker='o')  
    # plt.savefig('test.png')