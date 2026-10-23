\# ML Systems Labs - Alina Muradova (73686)



Individual lab work for the Introduction to Machine Learning Systems course.



\## Setup

python -m venv .venv

.venv\\Scripts\\activate

pip install -r requirements.txt



\## Structure

\- `lab01/` : environment setup and first system measurements (report in `lab01/report.md`)



\## Notes

\- Python 3.11.9 was used (the manual asks for 3.11.8).

\- `pyarrow` is pinned to 15.0.2 because `mlflow==2.14.1` requires `pyarrow<16`.

