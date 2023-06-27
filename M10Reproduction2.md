# Reprodukcja M10

Proponowany eksperyment zawiera test, który wykonuje 6 powtórzeń dla modelu Bagging:HoeffdingAdaptive i bazy danych npm.

Komendasłużąca do wywołania eksperymentu w katalogu ```/PythonModel/pythonmodel```.
```
poetry run py tester.py -e e1 -csv
```

Wyniki reprodukcji
| Iteration | MCC | G-Mean | Time-Elapsed |
| --------- | --- | ------ | -------------|
| 1 | 0.1798924747282669 | 0.5978432110465752 | 470.02228331565857 |
| 2 | 0.2230345102123441 | 0.6238677223025747 | 493.82949233055115 |
| 3 | 0.18556194297640816 | 0.6009745031747743 | 513.5790071487427 |
| 4 | 0.20024943244661086 | 0.6082375052798522 | 567.818318605423 |
| 5 | 0.20609527899319963 | 0.6118135459033062 | 603.3461449146271 |


Wyniki zamieszczone przez autorów
| MCC | G-Mean |
| --- | ------ |
| 0.157 | 0.5780 |

Wyniki opublikowane przez autorów pokrywają się z wynikami naszej reprodukcji.
