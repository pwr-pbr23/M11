# Zastosowanie Algorytmów Uczenia Maszynowego w Celu Detekcji Kodu Posiadającego Luki Bezpieczeństwa
PBR23M11

## [Paper #1](https://paperswithcode.com/paper/a-hierarchical-deep-neural-network-for)

## [LineVul paper](https://www.researchgate.net/publication/359402890_LineVul_A_Transformer-based_Line-Level_Vulnerability_Prediction)

## [Github Project](https://github.com/users/nadolnyjakub/projects/3)

## [Overleaf](https://www.overleaf.com/project/6401fb8af1232e4844c6bca3)

## [Colab Paper #1](https://colab.research.google.com/drive/1Mc7X-9XGZCP0gRZx4Xi17qpj9Hk-dvkU?usp=sharing)

## [Colab LineVul](https://colab.research.google.com/drive/1pbdftiX2dLcQbAuDYeSWUgnxHS8pLxrT?usp=sharing)

## [Colab output(wyuczone modele)](https://drive.google.com/drive/folders/1-1w-JAxryaX3ogvdInPaeUQYlSqoTk-B?usp=sharing)

# Autorzy

- Jakub Nadolny
- Kacper Mejsner
- Ivan Tarasiuk
- Łukasz Łosieczka


# Instrukcja reprodukcji LineVul

Zaleca się uruchomienie programu w środowisku chmurowym z dostępem do GPU.
Czas potrzebny na przeprowadzenie eksperymentów to około 2h na trenowanie i 15 minut na testy.
Zalecamy uruchomienie w Google Colab ze środowiskiem GPU.

1. Skolonować repozytorium, a konkretnie branch LineVul - to na nim obecnie pracujemy.
```
!git clone -b LineVul https://github.com/pwr-pbr23/M11.git
```
3. Przejść do katalogu: `%cd /content/M11/LineVul`
4. Zainstalować potrzebne pakiety:
```!pip install gdown
!pip install transformers
!pip install captum
!pip install torch
!pip install numpy
!pip install tqdm
!pip install pickle
!pip install sklearn
!pip install pandas
!pip install tokenizers
```

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

5. w pliku linevul_model.py aby użyć naszego modelu należy odkomentować i zakomentować odpowiednie linijki:

```
Linia 56
class Model(RobertaForSequenceClassification):   
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__(config=config)
        self.encoder = encoder
        self.tokenizer = tokenizer
        # self.classifier = Net(config) # nasz model
        self.classifier = RobertaClassificationHead(config) # model bazowy
        self.args = args
```

6. Uruchamianie trenowania:
```
%cd linevul
!python linevul_main.py \
  --output_dir=./saved_models \
  --model_type=roberta \
  --tokenizer_name=microsoft/codebert-base \
  --model_name_or_path=microsoft/codebert-base \
  --do_train \
  --do_test \
  --train_data_file=../data/big-vul_dataset/train.csv \
  --eval_data_file=../data/big-vul_dataset/val.csv \
  --test_data_file=../data/big-vul_dataset/test.csv \
  --epochs 1 \
  --block_size 256 \
  --train_batch_size 16 \
  --eval_batch_size 16 \
  --learning_rate 2e-5 \
  --max_grad_norm 1.0 \
  --evaluate_during_training \
  --seed 123456  2>&1 | tee train.log
%cd ..
```

7. Testy:
```
%cd linevul
!python linevul_main.py \
  --model_name=model.bin \
  --output_dir=./saved_models \
  --model_type=roberta \
  --tokenizer_name=microsoft/codebert-base \
  --model_name_or_path=microsoft/codebert-base \
  --do_test \
  --train_data_file=../data/big-vul_dataset/train.csv \
  --eval_data_file=../data/big-vul_dataset/val.csv \
  --test_data_file=../data/big-vul_dataset/test.csv \
  --block_size 256 \
  --eval_batch_size 256
%cd ..
```
# Wyniki badań

## Wyniki dla modelu LSTM i 1 epoce uczenia:
```
***** Running Test *****
  Num examples = 18864
    Batch size = 256
***** Test results *****
 test_accuracy = 0.9867
       test_f1 = 0.8709
test_precision = 0.9569
   test_recall = 0.7991
test_threshold = 0.5
```

## Wyniki dla modelu LSTM i 5 epokach
```
***** Running Test *****
   Num examples = 18864
     Batch size = 256
***** Test results *****
  test_accuracy = 0.9894
        test_f1 = 0.8991
       test_mcc = 0.8956
 test_precision = 0.9612
    test_recall = 0.8445
 test_threshold = 0.5
 ```
 
 ## Wyniki dla modelu Linear SVM i 4 epokach
```
***** Running Test *****
   Num examples = 18864
     Batch size = 256
***** Test results *****
  test_accuracy = 0.9891
        test_f1 = 0.8963
       test_mcc = 0.8925
 test_precision = 0.9560
    test_recall = 0.8436
 test_threshold = 0.5
 ```
