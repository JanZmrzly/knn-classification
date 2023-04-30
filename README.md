# Klasifikační model pro predikci

Tento program obsahuje dva skripty, které slouží k trénování a použití klasifikačního modelu pro predikci Segmentace na základě dat uložených v csv souboru. První skript natrénuje a uloží klasifikační model pomocí Train.csv datového souboru. Druhý skript následně načte natrénovaný model a provede inferenci na csv souboru zadaném v parametru příkazové řádky (viz sys.argv). Pokud csv soubor obsahuje sloupec "Segmentation", program vypíše na standardní výstup přesnost predikce ve formátu "accuracy: {počet_správných} / {počet_celkem}".

Cílem tohoto programu je umožnit trénování a použití klasifikačního modelu pro predikci Segmentace na základě csv souborů. Dále program umožňuje vypsat přesnost predikce na standardní výstup pro csv soubory obsahující sloupec "Segmentation".

## Instalace do virtuálního prostředí

```bash
pip install -r requirements.txt
```

### Hlavní použité knihovny

* [Pandas] (https://pandas.pydata.org/)
* [Seaborn] (https://seaborn.pydata.org/)
* [Matplotlib] (https://matplotlib.org/)
* [Scipy] (https://scipy.org/)
* [Scikit-learn] (https://scikit-learn.org/stable/)

## Popis souborů

* __knn.ipynb__ obsahuje prototypový kód, kde byla otestována funčnost,  obsahuje kód pro načtení dat, předzpracování a trénování modelu. Také obsahuje grafy, které budou uvedeny v dalším textu
* __train.py__ obsahuje část kódu ze souboru __knn.ipynb__. Tato část slouží pro natrénování dat, zobrazení výsledku trénování a uložení modelu.
* __load.py__ slouží pro validaci nad uloženým, natrénovaným modelem. Pokud je v datech sloupec Segmentaion, pak se zobrazí pouze "accuracy". Jinak je vytvořen soubor s predikcí dat a následné je uložen do slošky "results".

### Vstupní data

Vstupními daty jsou soubory s daty pro trénování modelu. Tyto soubory musí být ve formátu CSV.

### Výstupní data

Výstupem programu je natrénovaný model, soubor s daty a jejich predikcí nebo standartní výstup v terminálu.

## Příklad použití

```bash
python train.py data/Train.csv
```

```bash
python load.py data/Train.csv models/20230422_2035_kkn_model.joblib
```

## Metoda KNN


