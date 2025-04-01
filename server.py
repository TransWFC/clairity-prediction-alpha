from flask import Flask, jsonify
import requests
import datetime
import numpy as np
from pymongo import MongoClient
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configuración
MONGO_URI = "mongodb+srv://Sobeck:AlphaPrime%23.@alphaprime.4rg6f.mongodb.net/clairity?retryWrites=true&w=majority"
client = MongoClient(MONGO_URI)
db = client['clairity']
collection = db['sensordatas']

token = 'b38f9f302033008623310932a8ca0d6329fe9567'
cities = {
    "mexico": {"code": "mexico", "name": "Ciudad de México"},
    "monterrey": {"code": "monterrey", "name": "Monterrey"},
    "guadalajara": {"code": "guadalajara", "name": "Guadalajara"},
    "queretaro": {"code": "guanajuato", "name": "Querétaro", "alias_for": None}
}
days_history = 30

def get_historical_data(station_id, days):
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days)
    
    url = f'https://api.waqi.info/api/timeseries/{station_id}/?token={token}&start={start_date.date()}&end={end_date.date()}'
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and isinstance(data['data'], list):
                records = []
                for record in data['data']:
                    try:
                        if 'time' in record and 's' in record['time'] and 'iaqi' in record and 'pm25' in record['iaqi']:
                            dt = datetime.datetime.strptime(record['time']['s'], "%Y-%m-%d %H:%M:%S")
                            pm25 = float(record['iaqi']['pm25']['v'])
                            records.append({'date': dt, 'pm25': pm25})
                    except (KeyError, ValueError) as e:
                        continue
                
                if records:
                    df = pd.DataFrame(records)
                    df.sort_values('date', inplace=True)
                    return df
    except Exception as e:
        print(f"Error obteniendo datos históricos: {str(e)}")
    return None

def get_pm25_history_from_mongodb(days=30):
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days)
    
    records = list(collection.find(
        {"timestamp": {"$gte": start_date, "$lte": end_date}}, 
        {"_id": 0, "timestamp": 1, "AQI": 1}
    ))
    
    if records:
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        df['days_since'] = (df['timestamp'] - df['timestamp'].min()).dt.days
        return df
    return None


@app.route('/api/calidad-aire', methods=['GET'])
def get_air_quality():
    city_data = {}
    processed_cities = {}  # Para evitar procesar la misma ciudad múltiples veces
    
    for city_name, city_info in cities.items():
        if city_info.get('alias_for'):
            if city_info['alias_for'] in processed_cities:
                city_data[city_name] = processed_cities[city_info['alias_for']].copy()
                city_data[city_name]['name'] = city_info['name']
            continue
            
        try:
            print(f"\nProcesando ciudad: {city_name}")
            
            if city_name == "queretaro":
                  history_df = get_pm25_history_from_mongodb(days_history)
            else:
                url = f'https://api.waqi.info/feed/{city_info["code"]}/?token={token}'
                response = requests.get(url, timeout=10)
                
                if response.status_code != 200:
                    print(f"Error al obtener datos para {city_name}: {response.status_code}")
                    city_data[city_name] = {'name': city_info['name'], 'error': f"Error HTTP {response.status_code}"}
                    continue
                    
                data = response.json()
                station_id = data['data']['idx']
                print(f"ID de estación: {station_id}")
                
                current_pm25 = None
                current_time = None
                if 'iaqi' in data['data'] and 'pm25' in data['data']['iaqi'] and 'v' in data['data']['iaqi']['pm25']:
                    current_pm25 = float(data['data']['iaqi']['pm25']['v'])
                    current_time = datetime.datetime.strptime(data['data']['time']['s'], "%Y-%m-%d %H:%M:%S")
                    print(f"Dato actual - PM2.5: {current_pm25} en {current_time}")

            if current_pm25 is None:
                print("No se encontraron datos actuales de PM2.5")
                city_data[city_name] = {'name': city_info['name'], 'error': "No hay datos actuales de PM2.5"}
                continue
            
            forecast = []
            if city_name != "queretaro":
                forecast = data['data']['forecast']['daily']['pm25'] if 'forecast' in data['data'] and 'daily' in data['data']['forecast'] and 'pm25' in data['data']['forecast']['daily'] else []
            
            forecast_dates = [datetime.datetime.strptime(entry['day'], "%Y-%m-%d") for entry in forecast] if forecast else []
            forecast_values = [float(entry['avg']) for entry in forecast] if forecast else []
            
            history_df = get_historical_data(station_id, days_history) if city_name != "queretaro" else None
            
            if history_df is not None and not history_df.empty:
                history_df['days_since'] = (history_df['date'] - history_df['date'].min()).dt.days
                X = history_df['days_since'].values.reshape(-1, 1)
                y = history_df['pm25'].values
                
                model = make_pipeline(PolynomialFeatures(2), LinearRegression())
                model.fit(X, y)
                
                if forecast_dates:
                    future_days = np.arange(history_df['days_since'].max() + 1, 
                                          history_df['days_since'].max() + len(forecast_dates) + 1)
                    trend_prediction = model.predict(future_days.reshape(-1, 1))
                    combined_prediction = [(f + t*0.3)/1.3 for f, t in zip(forecast_values, trend_prediction)]
                else:
                    trend_prediction = []
                    combined_prediction = []
                
            result = {
                'name': city_info['name'],
                'history': history_df.to_dict('records') if history_df is not None else None,
                'current': {
                    'value': current_pm25,
                    'time': current_time.isoformat()
                },
                'forecast': {
                    'dates': [date.isoformat() for date in forecast_dates],
                    'values': forecast_values,
                    'trend': trend_prediction.tolist() if history_df is not None else [],
                    'combined': combined_prediction if history_df is not None else forecast_values
                }
            }
            city_data[city_name] = result
            
        except Exception as e:
            print(f"Error procesando ciudad {city_name}: {str(e)}")
            city_data[city_name] = {'name': city_info['name'], 'error': str(e)}
    
    return jsonify(city_data)

if __name__ == '__main__':
    app.run(port=8080, debug=True)
