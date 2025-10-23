from flask import Flask

app = Flask(__name__)
PORT = 5001 


if __name__ == '__main__':
    from dataset import datasets_bp
    from preprocessing import preprocessing_bp
    
    app.register_blueprint(datasets_bp, url_prefix='/')
    app.register_blueprint(preprocessing_bp, url_prefix='/')
    
    app.run(host='0.0.0.0', port=PORT)
