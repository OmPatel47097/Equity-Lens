import subprocess
import os

apps = ["blpo", "stock_trend_pred.", "backend"]

if __name__ == '__main__':
    app = input("Enter the file name of the application: ")
    if app == "backend":
        app_path = os.path.join('backend', 'app.py')
        subprocess.run(["python", app_path])
    elif app == "stock_trend_pred":
        app_path = os.path.join('prototypes', 'stock_trend_pred.py')
        subprocess.run(["streamlit", 'run', app_path])
    elif app == "blpo":
        app_path = os.path.join('prototypes', 'blpo.py')
        subprocess.run(["streamlit", 'run', app_path])
