import subprocess

def download_corpora():
    subprocess.run(["python","-m","textblob.download_corpora"])

if __name__ == "__main__":
    download_corpora()