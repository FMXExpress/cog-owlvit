import subprocess


def download_weights(url, dest):
    print("Downloading weights...")
    try:
        output = subprocess.check_output(["pget", "-x", url, dest])
    except subprocess.CalledProcessError as e:
        # If download fails, clean up and re-raise exception
        print(e.output)
        raise e
