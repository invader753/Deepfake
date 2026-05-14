import json
from detect import predict

def run_test():
    with open('real.jpeg', 'rb') as f:
        data = f.read()
    
    res = predict(data)
    print("=== RESULT ===")
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    run_test()
