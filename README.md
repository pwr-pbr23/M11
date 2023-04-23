# A Hierarchical Deep Neural Network for Detecting Lines of Codes with Vulnerabilities
PBR23M11

Paper #1: https://paperswithcode.com/paper/a-hierarchical-deep-neural-network-for

LineVul: https://www.researchgate.net/publication/359402890_LineVul_A_Transformer-based_Line-Level_Vulnerability_Prediction

Github Project: https://github.com/users/nadolnyjakub/projects/3

Overleaf: https://www.overleaf.com/project/6401fb8af1232e4844c6bca3

Colab: https://colab.research.google.com/drive/1Mc7X-9XGZCP0gRZx4Xi17qpj9Hk-dvkU?usp=sharing

Colab output: https://drive.google.com/drive/folders/1-1w-JAxryaX3ogvdInPaeUQYlSqoTk-B?usp=sharing

Colab LineVul: https://colab.research.google.com/drive/1pbdftiX2dLcQbAuDYeSWUgnxHS8pLxrT?usp=sharing

# Instrukcja reprodukcji

Zaleca się uruchomienie programu w środowisku chmurowym z dostępem do GPU.
Czas potrzebny na przeprowadzenie eksperymentów to około 2h na trenowanie i 15 minut na testy.
Zalecamy uruchomienie w Google Colab ze środowiskiem GPU.

1. Skolonować repozytorium, a konkretnie branch LineVul - to na nim obecnie pracujemy.
2. Przejść do katalogu: `%cd /content/M11/LineVul`
3. Zainstalować potrzebne pakiety:
```!pip install gdown
!pip install transformers
!pip install captum
!pip install torch
!pip install numpy
!pip install tqdm
!pip install pickle
!pip install sklearn
!pip install pandas
!pip install tokenizers```

4. Pobrać datasety. Uwaga, łącznie ważą ok. 20GB ale Google Colab daje dużą prędkość pobierania.
```
%cd data
%cd big-vul_dataset
!gdown https://drive.google.com/uc?id=1h0iFJbc5DGXCXXvvR6dru_Dms_b2zW4V
%cd ../..
```
```
%cd data
%cd big-vul_dataset
!gdown https://drive.google.com/uc?id=1ldXyFvHG41VMrm260cK_JEPYqeb6e6Yw
!gdown https://drive.google.com/uc?id=1yggncqivMcP0tzbh8-8Eu02Edwcs44WZ
%cd ../..
```
```
%cd data
%cd big-vul_dataset
!gdown https://drive.google.com/uc?id=10-kjbsA806Zdk54Ax8J3WvLKGTzN8CMX
%cd ../..
```

# Nasze obecne wyniki po podmianie modelu i 1 epoce trenowaia:
***** Running Test *****
  Num examples = 18864
    Batch size = 256
***** Test results *****
 test_accuracy = 0.9867
       test_f1 = 0.8709
test_precision = 0.9569
   test_recall = 0.7991
test_threshold = 0.5
