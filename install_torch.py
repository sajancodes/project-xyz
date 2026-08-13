import subprocess
result = subprocess.run(["python", "-m", "pip", "install", "torch", "--index-url", "https://pypi.org/simple", "--break-system-packages"], capture_output=True, text=True)
print("STDOUT:", result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
print("STDERR:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
print("Return code:", result.returncode)