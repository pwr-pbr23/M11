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

