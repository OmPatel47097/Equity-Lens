import subprocess
import os

apps = ["blpo", "stock_trend_pred.", "backend"]
blpo_app_path = os.path.join('prototypes', 'blpo.py')
hard_path = "./prototypes/blpo.py"
subprocess.run(["streamlit", 'run', hard_path])

# Note- For docker run purposes only
# app_path_2 = os.path.join('prototypes', 'blpo.py')
# print(app_path_2)
# subprocess.run(["python", "-m", "streamlit", 'run', app_path_2])

# if __name__ == '__main__':
#     app = input("Enter the file name of the application: ")
#     if app == "backend":
#         app_path = os.path.join('backend', 'app.py')
#         subprocess.run(["python", app_path])
#     elif app == "stock_trend_pred":
#         app_path = os.path.join('prototypes', 'stock_trend_pred.py')
#         subprocess.run(["streamlit", 'run', app_path])
#     elif app == "blpo":
#         app_path = os.path.join('prototypes', 'blpo.py')
#         subprocess.run(["streamlit", 'run', app_path])
