import pandas as pd
import pmdarima as pm
from sklearn.linear_model import LinearRegression

import pandas as pd
import pmdarima as pm
from sklearn.linear_model import LinearRegression

def load_data(file_path):
    '''loads the series dataset'''
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])
    return data

def stationarize_data_multiplicative(df, n_periods=365):
    '''decompose and stationarize the series using multiplicative method, accepts dataframe and n_periods for decomposistion
    produces stationarize dataframe and it's components'''
    dataframe = df.set_index('date').asfreq('D')
    result_decomposition = seasonal_decompose(dataframe['sales'], model='multiplicative', period=n_periods, extrapolate_trend='freq')
    y_diff = (dataframe['sales']/result_decomposition.seasonal/result_decomposition.trend).dropna()
    return y_diff, result_decomposition

def stationarize_data_additive(df, n_periods=365):
    '''decompose and stationarize the series using additive method if trend and seasonality values are zero, accepts dataframe and n_periods for decomposistion
    produces stationarize dataframe and it's components.'''
    dataframe = df.set_index('date').asfreq('D')
    result_decomposition = seasonal_decompose(dataframe['sales'], model='additive', period=n_periods, extrapolate_trend='freq')
    y_diff = (dataframe['sales']-result_decomposition.seasonal-result_decomposition.trend).dropna()
    return y_diff, result_decomposition
    
def auto_arima(dataframe, futures=365):
    '''making an auto arima model with the stationarized data, the futures are the number periods that the user wish to predict
    standard is one year. For multiplicative seasonal decomposed time series'''
    smodel = pm.auto_arima(dataframe, stationary=True, method='lbfgs', seasonal=True, m=12,
                       start_p=0, max_p=5, max_d=0, start_q=0, max_q=3,
                       start_P=0, max_P=2, max_D=0, start_Q=0, max_Q=3, 
                       trace=True, error_action='warn', suppress_warnings=True, n_fits=3, test='adf', random=True, maxiter=3)
    y_forec, conf_int  = smodel.predict(futures,return_conf_int=True,alpha=0.05)
    forecast = pd.DataFrame(y_forec, columns=['forecast'])
    conf_int = pd.DataFrame(conf_int, index=forecast.index, columns=['low','high'])
    return forecast, conf_int, smodel.aic()

def auto_arima_add(dataframe, futures=365):
    '''making an auto arima model with the stationarized data, the futures are the number periods that the user wish to predict
    standard is one year. Reduced fitting parameters for additive seasonal decomposed time series'''
    smodel = pm.auto_arima(dataframe, stationary=True, method='lbfgs', seasonal=True, m=12,
                       start_p=0, max_p=3, max_d=0, start_q=0, max_q=3,
                       start_P=0, max_P=2, max_D=0, start_Q=0, max_Q=3, 
                       trace=True, error_action='warn', suppress_warnings=True, n_fits=1, test='adf', random=True, maxiter=1)
    y_forec, conf_int  = smodel.predict(futures,return_conf_int=True,alpha=0.05)
    forecast = pd.DataFrame(y_forec, columns=['forecast'])
    conf_int = pd.DataFrame(conf_int, index=forecast.index, columns=['low','high'])
    return forecast, conf_int, smodel.aic()

def recompose_forecast_multiplicative(df_diff, df_forecast, confindence_interval, decomposition):
    '''Forecast the trend and seasonal components using Holt-Winters method, then Extrapolate the trend and seasonal components into the future
    . MULTIPLICATIVE METHOD'''
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    #holt winters method
    hw_trend = ExponentialSmoothing(trend, seasonal_periods=365, trend='mul', seasonal='mul').fit()
    hw_seasonal = ExponentialSmoothing(seasonal, seasonal_periods=365, trend=None, seasonal='mul').fit()
    #extrapolate into the future
    future_dates = pd.date_range(start=df_diff.index[-1]+timedelta(days=1), periods=365, freq='D')
    future_trend = hw_trend.predict(start=future_dates[0], end=future_dates[-1])
    future_seasonal = hw_seasonal.predict(start=future_dates[0], end=future_dates[-1])
    df_forecast['composed_forecast']=df_forecast['forecast']*future_trend*future_seasonal
    confindence_interval['low_composed']=confindence_interval['low']*future_trend*future_seasonal
    confindence_interval['high_composed']=confindence_interval['high']*future_trend*future_seasonal
    return df_forecast, confindence_interval

def recompose_forecast_additive(df_diff, df_forecast, confindence_interval, decomposition):
    '''Forecast the trend and seasonal components using Holt-Winters method, then Extrapolate the trend and seasonal components into the future
    . ADDITIVE METHOD'''
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    #holt winters method
    hw_trend = ExponentialSmoothing(trend, seasonal_periods=365, trend='add', seasonal=None).fit()
    hw_seasonal = ExponentialSmoothing(seasonal, seasonal_periods=365, trend=None, seasonal='add').fit()
    #extrapolate into the future
    future_dates = pd.date_range(start=df_diff.index[-1]+timedelta(days=1), periods=365, freq='D')
    future_trend = hw_trend.predict(start=future_dates[0], end=future_dates[-1])
    future_seasonal = hw_seasonal.predict(start=future_dates[0], end=future_dates[-1])
    df_forecast['composed_forecast']=df_forecast['forecast']+future_trend+future_seasonal
    confindence_interval['low_composed']=confindence_interval['low']+future_trend+future_seasonal
    confindence_interval['high_composed']=confindence_interval['high']+future_trend+future_seasonal
    return df_forecast, confindence_interval
    
def loop_through_arima(file_path):
    # load the data
    forecast_df = pd.DataFrame(columns=['date', 'store', 'item', 'forecast'])
    aic_df = pd.DataFrame(columns=['store', 'item', 'aic'])
    conf_int_df = pd.DataFrame(columns=['date', 'store', 'item', 'low', 'high'])
    dataframe = load_data(file_path)

    # loop over the store and item categories
    for store in dataframe['store'].unique():
        for item in dataframe['item'].unique():
            
            print(f'processing number {store} & {item}, starting multiplicative method')

            # extract data for the current store and item combination
            data = dataframe[(dataframe['store'] == store) & (dataframe['item'] == item)]
            
            try:  
                # stationarize the data
                y_diff, decomposition = stationarize_data_multiplicative(data, n_periods=365)

                # fit an ARIMA model to the stationarized data
                forecast, conf_int, aic = auto_arima(y_diff)

                # recompose the forecast to obtain predictions for the original data
                forecast, conf_int = recompose_forecast_multiplicative(y_diff, forecast, conf_int, decomposition)
            except ValueError:
                
                print('multiplicative method failed, additive method will be used')
                # stationarize the data
                y_diff, decomposition = stationarize_data_additive(data, n_periods=365)

                # fit an ARIMA model to the stationarized data
                forecast, conf_int, aic = auto_arima_add(y_diff)

                # recompose the forecast to obtain predictions for the original data
                forecast, conf_int = recompose_forecast_additive(y_diff, forecast, conf_int, decomposition)

            # add the forecast and AIC score to the respective dataframes
            conf_int_df = conf_int_df.append(pd.DataFrame({'date': forecast.index, 'store': store, 'item': item, 'low': conf_int['low_composed'].values, 'high': conf_int['high_composed'].values}))
            forecast_df = forecast_df.append(pd.DataFrame({'date': forecast.index, 'store': store, 'item': item, 'forecast': forecast['composed_forecast'].values}))
            aic_df = aic_df.append(pd.DataFrame({'store': [store], 'item': [item], 'aic': [aic]}))
            print(f"aic score for {store} & {item}: {aic}")
    
    return forecast_df, aic_df, conf_int_df

    
    
    

    
    
    