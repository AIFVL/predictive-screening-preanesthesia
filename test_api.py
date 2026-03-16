import urllib.request
import json
import sys

def test_api():
    try:
        # 1. Get models
        resp = urllib.request.urlopen('http://localhost:8000/models')
        models_data = json.loads(resp.read().decode('utf-8'))
        model_ids = [m['model_id'] for m in models_data['models']]
        print(f"Available models: {model_ids}")
        
        if not model_ids:
            print("No models available.")
            return
            
        model_id = model_ids[0]
        
        # 2. Get schema
        resp = urllib.request.urlopen(f'http://localhost:8000/predict/schema/{model_id}')
        schema = json.loads(resp.read().decode('utf-8'))
        print(f"\nSchema for {model_id}: {len(schema['features'])} features")
        
        # 3. Predict
        payload = {
            "model_id": model_id,
            "patient_data": {
                "Edad": 65,
                "ASA": 3,
                "severity_score_critical_proc": 1,
                "n_dx": 6,
                "SpO2": 95
            },
            "return_explanations": True
        }
        
        req = urllib.request.Request(
            'http://localhost:8000/predict',
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        resp = urllib.request.urlopen(req)
        result = json.loads(resp.read().decode('utf-8'))
        print(f"\nPredict result:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error during API test: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_api()
