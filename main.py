import subprocess
import sys

def main():
    print("Starting Credit Fair 2.0 Flask app...")
    subprocess.run([sys.executable, "App/app.py"])

if __name__ == "__main__":
    main()
