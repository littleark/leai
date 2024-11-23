import subprocess

def launch_streamlit():
    subprocess.run(["streamlit", "run", "streamlit_app.py"])

if __name__ == "__main__":
    launch_streamlit()
