# Systémová príručka

Systémová príručka na popis a spustenie všetkých kódov v tomto úložisku.

Repozitár obsahuje implementáciu troch modelov hlbokého učenia (LSTM, SlovakBERT, XLM-RoBERTa) pre automatickú detekciu toxického obsahu v slovenčine a skript na vysvetliteľnú analýzu (XAI) natrénovaných modelov pomocou metód LIME, SHAP, Integrated Gradients a Permutation Feature Importance.

---

## Požiadavky

Na spustenie skriptov je potrebné mať nainštalované nasledujúce knižnice:

```
torch==2.2.2
transformers==4.41.2
datasets==2.19.1
scikit-learn==1.4.2
scipy==1.13.0
numpy==1.26.4
pandas==2.2.2
lime==0.2.0.1
shap==0.45.1
matplotlib==3.8.4
captum==0.7.0
accelerate==0.29.3
fsspec==2024.3.1
```

Môžete ich nainštalovať pomocou príkazu:

```bash
pip install -r requirements.txt
```

Experimenty boli realizované na prostredí s GPU (CUDA 12.1). Na CPU je spustenie možné, avšak výrazne pomalšie.

---

## Trénovanie modelu LSTM (lstm.py)

Tento skript implementuje a trénuje obojsmernú rekurentnú neurónovú sieť LSTM ako baseline model pre detekciu toxicity v slovenčine. Dataset sa automaticky stiahne z platformy HuggingFace.

### Použitie

Spustite skript pomocou príkazu:

```bash
python3 lstm.py
```

### Funkcie skriptu

**Načítanie a rozdelenie dát:**
Skript načíta dataset TUKE-KEMT/toxic-sk z platformy HuggingFace pomocou knižnice `datasets`. Trénovaciu množinu rozdelí v pomere 90/10 na trénovaciu a validačnú pomocou stratifikovaného rozdelenia. Testovacia množina je použitá v podobe, v akej je publikovaná v datasete.

**Zostavenie slovníka:**
Z trénovacích dát sa zostaví slovník obsahujúci 20 000 najčastejších slov. Slová mimo slovníka sú nahradené tokenom `<UNK>`, prázdne pozície tokenom `<PAD>`.

**Definícia modelu:**
Model pozostáva z embedding vrstvy (dimenzia 128), dvoch vrstiev obojsmernej LSTM siete (256 skrytých jednotiek v každom smere), dropout regularizácie (hodnota 0,3) a plne prepojenej výstupnej vrstvy s dvoma neurónmi.

**Trénovanie:**
Model je trénovaný optimalizérom Adam s learning rate `1e-3`, batch size `64` a gradient clippingom s maximálnou normou `1,0`. Implementovaný je manuálny mechanizmus early stopping s patience = 3 na základe F1 skóre na validačnej množine. Maximálny počet epoch je 20. Vyššia hodnota batch size oproti transformerovým modelom je opodstatnená výrazne menším počtom parametrov LSTM siete, čo umožňuje efektívne využitie pamäte GPU bez rizika jej preťaženia.

**Vyhodnotenie a uloženie:**
Po trénovaní sa načítajú najlepšie váhy a model sa vyhodnotí na testovacej množine. Výsledky (Accuracy, Precision, Recall, F1) sa uložia do súboru `model_lstm_vysledky.txt`.

### Výstupy

- `model_lstm_best.pt` — váhy najlepšieho modelu
- `model_lstm_vysledky.txt` — výsledky na testovacej množine

---

## Fine-tuning modelu SlovakBERT (slovakbert.py)

Tento skript realizuje fine-tuning monolingválneho transformerového modelu SlovakBERT na úlohu binárnej klasifikácie toxicity. Model je predtrénovaný výhradne na slovenských textoch.

### Použitie

Spustite skript pomocou príkazu:

```bash
python3 slovakbert.py
```

### Funkcie skriptu

**Načítanie a rozdelenie dát:**
Skript načíta dataset TUKE-KEMT/toxic-sk z HuggingFace a rozdelí trénovaciu časť v pomere 90/10 na trénovaciu a validačnú množinu pomocou stratifikovaného rozdelenia.

**Tokenizácia:**
Text je tokenizovaný tokenizérom modelu `gerulata/slovakbert` s maximálnou dĺžkou sekvencie 128 tokenov a dynamickým paddingom pomocou `DataCollatorWithPadding`.

**Definícia modelu:**
Nad encoder časťou predtrénovaného modelu SlovakBERT je pridaná klasifikačná hlava pozostávajúca z dropout vrstvy a lineárnej vrstvy mapujúcej 768-dimenzionálnu reprezentáciu CLS tokenu na dva výstupy (toxický/neškodný).

**Trénovanie:**
Fine-tuning prebieha pomocou Hugging Face `Trainer` s optimalizérom AdamW, learning rate `1e-5`, batch size `16`, váhovou regularizáciou (weight decay) `0,1`, warmup obdobím `200` krokov, gradient clippingom s normou `1,0` a mixed-precision tréningom (fp16 pri dostupnosti GPU). Maximálny počet epoch je 10 s early stopping (patience = 3). Nižšia hodnota batch size oproti LSTM modelu je dôsledkom výrazne vyššieho počtu parametrov transformerovej architektúry (približne 125 miliónov), ktorý kladie vysoké nároky na pamäť GPU.

**Vyhodnotenie a uloženie:**
Po trénovaní sa najlepší model vyhodnotí na testovacej množine a výsledky (Accuracy, Precision, Recall, F1) sa uložia do textového súboru.

### Výstupy

- `model_slovakbert_simple/best/` — priečinok s najlepším modelom a tokenizérom
- `model_slovakbert_simple/vysledky.txt` — výsledky na testovacej množine

---

## Fine-tuning modelu XLM-RoBERTa (xlmr.py)

Tento skript realizuje fine-tuning multilingválneho transformerového modelu XLM-RoBERTa na úlohu binárnej klasifikácie toxicity. Model je predtrénovaný na textoch v 100 jazykoch vrátane slovenčiny.

### Použitie

Spustite skript pomocou príkazu:

```bash
python3 xlmr.py
```

### Funkcie skriptu

**Načítanie a rozdelenie dát:**
Skript načíta dataset TUKE-KEMT/toxic-sk z HuggingFace a rozdelí trénovaciu časť v pomere 90/10 na trénovaciu a validačnú množinu pomocou stratifikovaného rozdelenia.

**Tokenizácia:**
Text je tokenizovaný tokenizérom modelu `xlm-roberta-base` s maximálnou dĺžkou sekvencie 128 tokenov a dynamickým paddingom pomocou `DataCollatorWithPadding`.

**Definícia modelu:**
Nad encoder časťou predtrénovaného modelu XLM-RoBERTa je pridaná klasifikačná hlava mapujúca 768-dimenzionálnu reprezentáciu CLS tokenu na dva výstupy.

**Trénovanie:**
Fine-tuning prebieha pomocou Hugging Face `Trainer` s optimalizérom AdamW, learning rate `2e-5`, batch size `16`, váhovou regularizáciou (weight decay) `0,1`, warmup obdobím `200` krokov, gradient clippingom s normou `1,0` a mixed-precision tréningom (fp16 pri dostupnosti GPU). Maximálny počet epoch je 10 s early stopping (patience = 3). Pre konzistentnosť porovnania s modelom SlovakBERT je použitá rovnaká hodnota batch size 16.

**Vyhodnotenie a uloženie:**
Po trénovaní sa najlepší model vyhodnotí na testovacej množine a výsledky (Accuracy, Precision, Recall, F1) sa uložia do textového súboru.

### Výstupy

- `model_xlmr/best/` — priečinok s najlepším modelom a tokenizérom
- `model_xlmr/vysledky.txt` — výsledky na testovacej množine

---

## XAI analýza natrénovaných modelov (XAI.py)

Tento skript realizuje vysvetliteľnú analýzu všetkých troch natrénovaných modelov pomocou štyroch XAI metód: LIME, SHAP, Integrated Gradients a Permutation Feature Importance. Doplnkovo sa generujú matice zámen a porovnávací graf metrík. Skript je potrebné spustiť až po natrénovaní všetkých troch modelov.

### Požiadavky

Pred spustením XAI analýzy musia existovať nasledujúce súbory a priečinky:

- `model_lstm_best.pt` — uložený LSTM model
- `model_slovakbert_simple/best/` — uložený SlovakBERT model
- `model_xlmr/best/` — uložený XLM-RoBERTa model

### Použitie

Spustite skript pomocou príkazu:

```bash
python3 XAI.py
```

### Funkcie skriptu

**Načítanie modelov:**
Skript načíta váhy všetkých troch natrénovaných modelov. Pre transformerové modely sa načíta aj tokenizér z príslušného priečinka.

**Definícia ilustračných komentárov:**
Lokálne XAI analýzy (LIME, SHAP, Integrated Gradients) sú realizované na šiestich ilustračných komentároch (troch toxických a troch neškodných) konštruovaných tak, aby reprezentovali typické vzory v datasete. Permutation Feature Importance je naopak realizovaná na celej testovacej množine.

**LIME analýza (`run_lime`):**
Pre každý model a každý komentár sa vygeneruje lokálne vysvetlenie pomocou knižnice `lime.lime_text`. Funkcia predikcie je obalená tak, aby pracovala s príslušnou reprezentáciou vstupu (indexy slov pre LSTM, tokeny pre transformery). Výsledky sú vizualizované ako stĺpcové grafy s váhami jednotlivých tokenov.

**SHAP analýza (`run_shap`):**
Pre každý model sa vypočítajú Shapleyho hodnoty pomocou knižnice `shap`. Pre LSTM je použitý `KernelExplainer`, pre transformerové modely `PartitionExplainer`. Výsledky sú vizualizované ako horizontálne stĺpcové grafy.

**Integrated Gradients (`run_ig`):**
Pre transformerové modely (SlovakBERT, XLM-RoBERTa) sa vypočítajú atribúcie tokenov pomocou metódy Integrated Gradients implementovanej v knižnici `captum`. Atribúcie sú distribuované medzi subword tokeny generované tokenizérom. Implementácia využíva spoločnú RoBERTa-based architektúru oboch modelov (atribút `model.roberta`).

**Permutation Feature Importance (`run_pfi_lstm`, `run_pfi_transformer`):**
Pre každý model sa zmeria pokles F1 skóre pri permutácii hodnôt na každej tokenovej pozícii (pozície 0–14) cez celú testovaciu množinu. Analýza je obmedzená na prvých 15 pozícií, keďže väčšina komentárov v datasete je krátka a tieto pozície obsahujú najrelevantnejšie informácie. Pre zaistenie reprodukovateľnosti je na začiatku každej PFI funkcie nastavený random seed `42`. Výsledky sú vizualizované ako stĺpcové grafy dôležitosti jednotlivých pozícií.

**Matice zámen (`plot_cm`):**
Pre každý model sa na celej testovacej množine vypočíta a vizualizuje matica zámen.

**Porovnanie metrík (`plot_porovnanie`):**
Vygeneruje sa stĺpcový graf porovnávajúci Accuracy, Precision, Recall a F1 skóre všetkých troch modelov.

**Poznámka:** Hodnoty metrík vo funkcii `plot_porovnanie` sú zapísané pevne v kóde. Pred spustením XAI analýzy ich aktualizujte podľa skutočných výsledkov v súboroch `model_lstm_vysledky.txt`, `model_slovakbert_simple/vysledky.txt` a `model_xlmr/vysledky.txt`.

### Výstupy

Skript uloží PNG obrázky grafov do priečinka `figures/` (vytvorí sa automaticky):

- `lime_lstm.png`, `lime_slovakbert.png`, `lime_xlm_roberta.png`
- `shap_lstm.png`, `shap_slovakbert.png`, `shap_xlm_roberta.png`
- `ig_slovakbert.png`, `ig_xlm_roberta.png`
- `pfi_lstm.png`, `pfi_slovakbert.png`, `pfi_xlm_roberta.png`
- `cm_lstm.png`, `cm_slovakbert.png`, `cm_xlm_roberta.png` — matice zámen
- `porovnanie_metrik.png` — porovnanie metrík všetkých modelov

---

## Odporúčané poradie spustenia

```bash
# 1. Inštalácia závislostí
pip install -r requirements.txt

# 2. Trénovanie modelov
python3 lstm.py
python3 slovakbert.py
python3 xlmr.py

# 3. Aktualizácia hodnôt metrík vo funkcii plot_porovnanie v XAI.py
#    podľa skutočných výsledkov z vysledky.txt súborov

# 4. XAI analýza (až po dokončení trénovania všetkých modelov)
python3 XAI.py
```

---

## Reprodukovateľnosť

Všetky experimenty používajú pevný random seed `42` nastavený pre knižnice `torch`, `numpy` aj Hugging Face `Trainer` (vrátane permutácií v PFI analýze). Výsledky sa môžu mierne líšiť v závislosti od verzie knižníc a hardvéru (CPU vs. GPU, typ GPU).
