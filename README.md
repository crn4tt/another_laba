```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install flet plotly numpy scipy kaleido nuitka

python -m nuitka --standalone --onefile --include-package=plotly --include-package-data=plotly --include-package=kaleido --include-package-data=kaleido main.py

.\main.exe
```