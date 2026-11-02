\# Manual check of 50 random rows



Sample: `results/audit\_sample\_50.csv` (df.sample(n=50, random\_state=42) on the raw data).

I checked every row against the UCI Adult data dictionary.



\- Impossible values: 0 (age, hours-per-week, capital-gain/loss and education-num are all in range).

\- education and education-num agree in every row (e.g. HS-grad = 9, Bachelors = 13, Doctorate = 16).

\- relationship agrees with sex (every Wife is Female, every Husband is Male).

\- No row has both capital-gain > 0 and capital-loss > 0.

\- Missing values: 3 rows (workclass and occupation missing together in 2 rows, native-country missing in 1 row).

\- Unusual but valid: hours-per-week = 4 (row 3375), hours-per-week = 84 (row 3136), age 40 "Own-child" (row 7155). I kept them because they are inside the allowed ranges.



Duplicate-conflict audit (whole dataset): 5 groups of identical feature rows with different income labels, 10 rows in total.

