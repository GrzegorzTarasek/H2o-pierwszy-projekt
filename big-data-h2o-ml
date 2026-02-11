# H2o-pierwszy-projekt

```{r}
library(h2o)
h2o.init(nthreads = -1,  # Liczba wątków: -1 oznacza użycie wszystkich dostępnych rdzeni
         max_mem_size = "8G") # Maksymalna ilość pamięci przydzielonej dla H2O
```


```{r}
loan_csv <- "https://raw.githubusercontent.com/h2oai/app-consumer-loan/master/data/loan.csv"
data <- h2o.importFile(loan_csv)
```
```{r}
data
```

```{r}
data$bad_loan <- as.factor(data$bad_loan)  # kodujemy zmienną binarną jako faktor
h2o.levels(data$bad_loan)  # opcjonalnie: sprawdzamy poziomy faktora ('0' i '1')
```


```{r}
splits <- h2o.splitFrame(data = data, 
                         ratios = c(0.7, 0.15),  # podział: 70%, 15%, 15%
                         seed = 1)
train <- splits[[1]]
valid <- splits[[2]]
test <- splits[[3]]
```

```{r}
nrow(train)
nrow(valid)
nrow(test)
```

```{r}
y <- "bad_loan"
x <- setdiff(names(data), c(y))
print(x)
```


### Gradient Boosting Machine



Gradient Boosting Machine działa na drzewach decyzyjnych, które są prostymi modelami decyzyjnymi, działającymi jak zestaw pytań (np. "Czy wiek jest większy niż 30 lat?"). Jednak pojedyncze drzewo jest słabe, dlatego GBM buduje wiele drzew, jedno po drugim, w taki sposób, że każde kolejne drzewo poprawia błędy poprzednich. Algorytm stara się zoptymalizować błędy (funkcję straty) poprzez dodawanie nowych drzew, które "skupiają się" na przypadkach, które zostały źle sklasyfikowane przez poprzednie drzewa. W ten sposób GBM stopniowo tworzy bardzo mocny model.




To lista, która zawiera parametry dla modelu Gradient Boosting Machine (GBM):
- **`learn_rate`**: wektor z wartościami `c(0.01, 0.1)`, które określają szybkość uczenia modelu.
- **`max_depth`**: wektor `c(3, 5, 9)`, który ustala maksymalną głębokość drzewa decyzyjnego.
- **`sample_rate`**: wektor `c(0.8, 1.0)`, który ustala proporcję próbek używanych w treningu.
- **`col_sample_rate`**: wektor `c(0.2, 0.5, 1.0)`, który ustala procent cech wybieranych w każdym drzewie.

```{r}
gbm_params1 <- list(learn_rate = c(0.01, 0.1),
                    max_depth = c(3, 5, 9),
                    sample_rate = c(0.8, 1.0),
                    col_sample_rate = c(0.2, 0.5, 1.0))
```

To lista, która określa kryteria poszukiwania najlepszych parametrów dla modelu:
- **`strategy`**: ustawione na `"RandomDiscrete"`, co oznacza, że poszukiwania są realizowane losowo, w sposób dyskretny.
- **`max_runtime_secs`**: ustawione na `120`, co oznacza, że proces poszukiwania może trwać maksymalnie 120 sekund.

```{r}
search_criteria <- list(strategy = "RandomDiscrete", 
                        max_runtime_secs = 120)
```
Zmienna, która tworzy nową siatkę parametrów dla modelu GBM. Składa się z:
- **`x = x, y = y`**: Zmienne objaśniające (`x`) oraz zmienna celu (`y`).
- **`grid_id`**: identyfikator tej siatki, ustawiony na `"gbm_grid1"`.
- **`training_frame`**: ramka danych do treningu, zdefiniowana jako `train`.
- **`validation_frame`**: ramka danych do walidacji, zdefiniowana jako `valid`.
- **`ntrees`**: liczba drzew w modelu, ustawiona na `100`.
- **`seed`**: ziarno generatora liczb losowych, ustawione na `1`.
- **`hyper_params`**: odnosi się do zmiennych parametrów hyper, czyli `gbm_params1`.
- **`search_criteria`**: odnosi się do kryteriów wyszukiwania, czyli `search_criteria`.
```{r}
# Tworzymy nową siatkę z tymi samymi danymi
gbm_grid1 <- h2o.grid("gbm", x = x, y = y,
                         grid_id = "gbm_grid1",  # Nowa siatka z nowym ID
                         training_frame = train,
                         validation_frame = valid,
                         ntrees = 100,
                         seed = 1,
                         hyper_params = gbm_params1,
                         search_criteria = search_criteria)

```
Zmienna, która zawiera wyniki uzyskane po przeprowadzeniu poszukiwań najlepszych parametrów. Składa się z:
- **`h2o.getGrid`**: funkcja służąca do pobrania wyników grid search dla określonego `grid_id` (tutaj `"gbm_grid1"`).
- **`sort_by`**: kryterium sortowania wyników, ustawione na `"auc"` (area under curve), czyli miara jakości modelu.
- **`decreasing`**: ustawione na `TRUE`, co oznacza, że wyniki będą posortowane malejąco.
```{r}
gbm_gridperf1 <- h2o.getGrid(grid_id = "gbm_grid1", 
                             sort_by = "auc", 
                             decreasing = TRUE)
print(gbm_gridperf1)
```

```{r}
gbm_params2 <- list(learn_rate = seq(0.01, 0.1, 0.01),
                    max_depth = seq(2, 10, 1),
                    sample_rate = seq(0.5, 1.0, 0.1),
                    col_sample_rate = seq(0.1, 1.0, 0.1))
search_criteria2 <- list(strategy = "RandomDiscrete", 
                         max_models = 36)
```

```{r}
gbm_grid2 <- h2o.grid("gbm", x = x, y = y,
                      grid_id = "gbm_grid2",
                      training_frame = train,
                      validation_frame = valid,
                      ntrees = 100,
                      seed = 1,
                      hyper_params = gbm_params2,
                      search_criteria = search_criteria2)

gbm_gridperf2 <- h2o.getGrid(grid_id = "gbm_grid2", 
                             sort_by = "auc", 
                             decreasing = TRUE)
print(gbm_gridperf2)
```

```{r}
gbm_params <- list(learn_rate = seq(0.1, 0.3, 0.01),  # zaktualizowane
                   max_depth = seq(2, 10, 1),
                   sample_rate = seq(0.9, 1.0, 0.05),  # zaktualizowane
                   col_sample_rate = seq(0.1, 1.0, 0.1))
search_criteria <- list(strategy = "RandomDiscrete", 
                         max_runtime_secs = 60)  # zaktualizowane


gbm_grid <- h2o.grid("gbm", x = x, y = y,
                      grid_id = "gbm_grid2",
                      training_frame = train,
                      validation_frame = valid,
                      ntrees = 100,
                      seed = 1,
                      hyper_params = gbm_params,
                      search_criteria = search_criteria2)

gbm_gridperf <- h2o.getGrid(grid_id = "gbm_grid2", 
                             sort_by = "auc", 
                             decreasing = TRUE)
print(gbm_gridperf)
```

```{r}
best_gbm_model_id <- gbm_gridperf@model_ids[[1]]
best_gbm <- h2o.getModel(best_gbm_model_id)
```

```{r}
best_gbm_perf <- h2o.performance(model = best_gbm, 
                                 newdata = test)
h2o.auc(best_gbm_perf)
```




### Deep Learning



Deep Learning działa na podstawie sztucznych sieci neuronowych, które są inspirowane tym, jak działa ludzki mózg. Sieć składa się z warstw neuronów, które analizują dane w sposób stopniowy. Na przykład, jeśli uczymy sieć rozpoznawania obrazów, pierwsza warstwa może wykrywać proste cechy, jak krawędzie, druga bardziej złożone, jak kąty, a trzecia – całe obiekty. Sieci neuronowe "uczą się" na podstawie przykładów, a algorytm stopniowo poprawia się, dostosowując swoje parametry (wagi), by lepiej rozpoznawać wzorce w danych.



Zmienna ta zawiera wektor różnych funkcji aktywacji, które mogą być użyte w modelu Deep Learning. Funkcje aktywacji wpływają na sposób, w jaki sieć neuronowa przetwarza dane wejściowe w swoich warstwach. Dostępne opcje to:
- **`"Rectifier"`**: Najczęściej stosowana funkcja aktywacji w sieciach neuronowych (ReLU).
- **`"RectifierWithDropout"`**: Wersja funkcji ReLU, która dodatkowo stosuje technikę dropout, aby zapobiec przeuczeniu.
- **`"Maxout"`**: Alternatywna funkcja aktywacji, która może pomóc w modelowaniu bardziej złożonych zależności.
- **`"MaxoutWithDropout"`**: Wersja Maxout, która również wykorzystuje technikę dropout.

Zmienna zawiera wektor wartości, które mogą być użyte do regularyzacji L1. Regularyzacja L1 dodaje karę do funkcji kosztu modelu, aby zapobiec przeuczeniu, zmieniając wartości wag. Wartości w tym wektorze to:
- **`c(0, 0.00001, 0.0001, 0.001, 0.01, 0.1)`**: Wartości regularyzacji L1, gdzie 0 oznacza brak regularyzacji.

Zmienna zawiera wektor wartości, które mogą być użyte do regularyzacji L2. Podobnie jak regularyzacja L1, L2 również przeciwdziała przeuczeniu, ale w inny sposób, nakładając karę na kwadraty wag. Wartości w tym wektorze to:
- **`c(0, 0.00001, 0.0001, 0.001, 0.01, 0.1)`**: Wartości regularyzacji L2, podobnie jak w `l1_opt`.


```{r}
activation_opt <- c("Rectifier", "RectifierWithDropout", "Maxout", "MaxoutWithDropout")
l1_opt <- c(0, 0.00001, 0.0001, 0.001, 0.01, 0.1)
l2_opt <- c(0, 0.00001, 0.0001, 0.001, 0.01, 0.1)
```

Jest to lista, która zawiera hiperparametry, które będą testowane w procesie trenowania modelu. Zawiera następujące elementy:
- **`activation`**: Wybór jednej z funkcji aktywacji z wektora `activation_opt`.
- **`l1`**: Wybór wartości regularyzacji L1 z wektora `l1_opt`.
- **`l2`**: Wybór wartości regularyzacji L2 z wektora `l2_opt`.

```{r}
hyper_params <- list(activation = activation_opt,
                     l1 = l1_opt,
                     l2 = l2_opt)
```

Zmienna, która definiuje siatkę hiperparametrów dla modelu Deep Learning. To część procesu tzw. grid search, który umożliwia znalezienie najlepszych parametrów modelu. Składa się z:
- **`x = x, y = y`**: Zmienne objaśniające (`x`) i zmienna celu (`y`), które są wykorzystywane do trenowania modelu.
- **`grid_id`**: Identyfikator tej siatki, ustawiony na `"dl_grid"`.
- **`training_frame`**: Ramka danych wykorzystywana do treningu modelu, zdefiniowana jako `train`.
- **`validation_frame`**: Ramka danych wykorzystywana do walidacji modelu, zdefiniowana jako `valid`.
- **`seed`**: Ziarno generatora liczb losowych, aby zapewnić powtarzalność wyników, ustawione na `1`.
- **`hidden`**: Określenie liczby ukrytych warstw w sieci neuronowej (tutaj ustawiono `c(10)`).
- **`hyper_params`**: Zmienna, która odnosi się do hiperparametrów, czyli `hyper_params`.
- **`search_criteria`**: Określa kryteria wyszukiwania, jak wcześniej opisano.

```{r}
dl_grid <- h2o.grid("deeplearning", x = x, y = y,
                    grid_id = "dl_grid",
                    training_frame = train,
                    validation_frame = valid,
                    seed = 1,
                    hidden = c(10,10),
                    hyper_params = hyper_params,
                    search_criteria = search_criteria)
```

Zmienna przechowująca wyniki wyszukiwania najlepszych parametrów w siatce modelu. Obejmuje wyniki grid search z posortowanymi wartościami:
- **`h2o.getGrid`**: Funkcja pobierająca wyniki grid search dla podanego `grid_id` ("dl_grid").
- **`sort_by`**: Określa, według jakiej miary wyniki mają być posortowane (w tym przypadku `"auc"`, czyli area under curve).
- **`decreasing`**: Określa, czy sortowanie ma odbywać się w porządku malejącym (ustawione na `TRUE`, aby najlepiej oceniane modele były na górze listy).

```{r}
dl_gridperf <- h2o.getGrid(grid_id = "dl_grid", 
                           sort_by = "auc", 
                           decreasing = TRUE)
print(dl_gridperf)
```

Zmienna zawierająca identyfikator najlepszego modelu, wybranego na podstawie wyników `dl_gridperf`. Jest to pierwszy model z posortowanej siatki, tj. model o najwyższej wartości AUC:
- **`dl_gridperf$model_ids[[1]]`**: Wybór najlepszego modelu na podstawie posortowanej siatki wyników.

```{r}
best_dl_model_id <- dl_gridperf@model_ids[[1]]
best_dl <- h2o.getModel(best_dl_model_id)
```

Zmienna zawierająca szczegółową ocenę wydajności najlepszego modelu na zbiorze testowym. Jest to wynik zastosowania modelu na nowych danych (w tym przypadku na zbiorze `test`):
- **`h2o.performance`**: Funkcja, która oblicza wydajność modelu na nowym zbiorze danych.

```{r}
best_dl_perf <- h2o.performance(model = best_dl, 
                                newdata = test)
h2o.auc(best_dl_perf)
```



### Deep Learning klasyfikacja wieloraka



W drugim przykładzie zastosowano algorytm Deep Learning do klasyfikacji wieloklasowej na zbiorze danych MNIST, zawierającym obrazy cyfr od 0 do 9.




```{r}
# Załaduj dane z innego źródła, aby uniknąć konfliktów z wcześniejszymi danymi
train_file_new <- "https://h2o-public-test-data.s3.amazonaws.com/bigdata/laptop/mnist/train.csv.gz"
test_file_new <- "https://h2o-public-test-data.s3.amazonaws.com/bigdata/laptop/mnist/test.csv.gz"
train_new <- h2o.importFile(train_file_new)
test_new <- h2o.importFile(test_file_new)

```

```{r}
y_new <- "C785"  # zmieniamy nazwę zmiennej celu, aby uniknąć konfliktów
x_new <- setdiff(names(train_new), y_new)  # zmiana zmiennej predyktora

```


```{r}
# Ponownie kodujemy zmienną celu jako faktor
train_new[,y_new] <- as.factor(train_new[,y_new])
test_new[,y_new] <- as.factor(test_new[,y_new])

```

**`dl_fiti_new`**:
   - To zmienna, która przechowuje **model Deep Learning** po jego wytrenowaniu. Model ten jest tworzony za pomocą funkcji `h2o.deepLearning()`, a po jego wytrenowaniu przechowywane są w niej wszystkie wyniki oraz parametry modelu.

**`h2o.deepLearning()`**:
   - Funkcja z biblioteki H2O, która tworzy i trenuje model głębokiego uczenia (sieci neuronowej). 
   
   Parametry funkcji:
   - **`x = x_new`**: Określa zmienne objaśniające (cechy), które będą wykorzystywane do trenowania modelu. **`x_new`** to zbiór danych wejściowych (cechy).
   - **`y = y_new`**: Określa zmienną celu, którą model będzie starał się przewidywać. **`y_new`** to zmienna, którą model ma przewidywać na podstawie danych z **`x_new`**.
   - **`training_frame = train_new`**: Zbiór danych, na którym model będzie trenowany. **`train_new`** zawiera dane wejściowe i wyjściowe do nauki.
   - **`model_id = "dl_fiti_new"`**: Unikalny identyfikator modelu. Dzięki temu identyfikatorowi można odwołać się do modelu w późniejszych etapach analizy.
   - **`hidden = c(20, 20)`**: Struktura modelu – liczba warstw ukrytych oraz liczba neuronów w każdej warstwie. W tym przypadku model ma 2 warstwy ukryte, każda z 20 neuronami.


```{r}
# Pierwszy model - bez wcześniejszych zmian
dl_fit1_new <- h2o.deeplearning(x = x_new,
                                 y = y_new,
                                 training_frame = train_new,
                                 model_id = "dl_fit1_new",
                                 hidden = c(20, 20),
                                 seed = 1)

```
Ostrzeżenie informuje, że pewne kolumny zostały **odrzucone**, ponieważ były **stałe** lub zawierały **błędy**.



W tym kodzie zmieniono liczbę **epok** na 50, co oznacza, że model będzie trenowany przez więcej cykli (iteracji). Większa liczba epok może poprawić dokładność modelu, ale zwiększa czas treningu. Ponadto, włączono parametr **`stopping_rounds = 0`**, co oznacza, że **nie będzie używane wczesne zatrzymywanie**, czyli model nie zostanie przerwany, nawet jeśli nie poprawia się jego wydajność po kilku epokach.


```{r}
# Drugi model - z większą liczbą epok
dl_fit2_new <- h2o.deeplearning(x = x_new,
                                 y = y_new,
                                 training_frame = train_new,
                                 model_id = "dl_fit2_new",
                                 epochs = 50,
                                 hidden = c(20, 20),
                                 stopping_rounds = 0,  # wyłączenie wczesnego zatrzymywania
                                 seed = 1)

```

W tym kodzie dodano **kroswalidację** do modelu. Parametr **`nfolds = 3`** oznacza, że dane zostaną podzielone na 3 części (foldy), a model będzie trenowany na 2 z nich, a testowany na pozostałej części. Takie podejście pozwala na lepszą ocenę wydajności modelu, ponieważ jest testowany na różnych podzbiorach danych.

Dodatkowo, wprowadzone zostały:
- **`score_interval = 1`**: Co oznacza, że ocena modelu będzie odbywać się co 1 epokę.
- **`stopping_rounds = 5`**: Oznacza, że jeśli przez 5 kolejnych ocen (epok) nie nastąpi poprawa lepsza niż określona w `stopping_tolerance`, trening zostanie przerwany wcześniej.
- **`stopping_metric = "misclassification"`**: Określa, że miarą wydajności modelu będzie liczba błędów klasyfikacji.
- **`stopping_tolerance = le-3`**: Określa tolerancję zatrzymywania modelu, jeśli poprawa wydajności będzie mniejsza niż 0.001.


```{r}
# Trzeci model z użyciem kroswalidacji
dl_fit3_new <- h2o.deeplearning(x = x_new,
                                 y = y_new,
                                 training_frame = train_new,
                                 model_id = "dl_fit3_new",
                                 epochs = 50,
                                 hidden = c(20, 20),
                                 nfolds = 3,                            # używane do wczesnego zatrzymywania
                                 score_interval = 1,                    # używane do wczesnego zatrzymywania
                                 stopping_rounds = 5,                   # używane do wczesnego zatrzymywania
                                 stopping_metric = "misclassification", # używane do wczesnego zatrzymywania
                                 stopping_tolerance = 1e-3,             # używane do wczesnego zatrzymywania
                                 seed = 1)


```

- Funkcja **`h2o.performance()`** służy do oceny wydajności modelu na danych testowych (**`test_new`**). Oblicza metryki, takie jak AUC, dokładność, log loss itp.
  

```{r}
# Sprawdzamy wyniki dla wszystkich trzech modeli
dl_perf1_new <- h2o.performance(model = dl_fit1_new, newdata = test_new)
dl_perf2_new <- h2o.performance(model = dl_fit2_new, newdata = test_new)
dl_perf3_new <- h2o.performance(model = dl_fit3_new, newdata = test_new)

# Wyświetlamy MSE dla każdego modelu
h2o.mse(dl_perf1_new)

h2o.mse(dl_perf2_new) 

h2o.mse(dl_perf3_new)

```
Funkcja h2o.scoreHistory(dl_fit3_new) zwraca historię treningu modelu, pokazując, jak zmieniały się metryki (np. błąd klasyfikacji, logloss, RMSE) w kolejnych epokach. Dzięki temu można ocenić, czy model się uczy, czy doszło do przeuczenia oraz w którym momencie trening mógł zostać zatrzymany.


```{r}
# Historia wyników modelu 3
h2o.scoreHistory(dl_fit3_new)

```
W tabeli widzimy następujące kolumny:

1. **`timestamp`**:
   - Czas, w którym została zarejestrowana dana epoka treningu.
   
2. **`duration`**:
   - Czas trwania danej epoki treningu, pokazuje, jak długo trwało przetwarzanie danej iteracji.

3. **`training_speed`**:
   - Prędkość trenowania, mierzona w liczbie przetworzonych obserwacji na sekundę (np. `80760 obs/sec`).

4. **`epochs`**:
   - Numer bieżącej epoki (np. 1, 2, 3, ...). Określa, jak wiele razy model przechodzi przez cały zbiór danych treningowych.

5. **`iterations`**:
   - Liczba wykonanych iteracji (podzielonych na partie) w danej epoce. W zależności od liczby próbek, jedna epoka może składać się z wielu iteracji.

6. **`samples`**:
   - Liczba próbek (obserwacji) przetworzonych do danej epoki.

7. **`training_rmse`**:
   - **RMSE (Root Mean Squared Error)**: Miara błędu modelu na danych treningowych. Niższe wartości wskazują na lepszą jakość modelu.

8. **`training_logloss`**:
   - **Log loss**: Miara błędu, która ocenia, jak dobrze model przewiduje prawdopodobieństwa. Niższa wartość wskazuje na lepsze prognozy.

9. **`training_r2`**:
   - **R²** (coefficient of determination): Mierzy, jak dobrze model wyjaśnia zmienność zmiennej celu. Wyższe wartości oznaczają lepsze dopasowanie modelu.

10. **`training_classification_error`**:
   - Błąd klasyfikacji, czyli odsetek błędnie sklasyfikowanych próbek. Niższa wartość oznacza lepszy model klasyfikacyjny.

11. **`training_auc` (AUC - Area Under Curve)**:
   - Mierzy jakość modelu w zadaniach klasyfikacji. Wyższa wartość AUC oznacza, że model lepiej rozróżnia klasy.

12. **`training_pr_auc` (Precision-Recall AUC)**:
    - Miara jakości modelu klasyfikacyjnego, szczególnie dla danych z niezrównoważonymi klasami. Wyższa wartość oznacza lepszą precyzję i czułość modelu.
    
    
    
    
    
    
    
Macierz konfuzji przedstawia wyniki klasyfikacji modelu, pokazując, jak dobrze model przewiduje poszczególne klasy:

1. **Wiersze**: Reprezentują prawdziwe etykiety (rzeczywiste klasy w zbiorze testowym).
2. **Kolumny**: Reprezentują przewidywane etykiety przez model.
3. **Komórki**: Liczba przypadków, w których prawdziwa etykieta z wiersza została sklasyfikowana jako przewidywana etykieta z kolumny.

- **Wysokie liczby na głównej przekątnej** (np. `984`, `1083`, `947`) oznaczają poprawne klasyfikacje.
- **Niższe liczby poza główną przekątną** wskazują błędnie sklasyfikowane przypadki.


```{r}
# Macierz konfuzji dla modelu 3
h2o.confusionMatrix(dl_fit3_new)

```

Wykres przedstawia **błąd klasyfikacji** modelu **`dl_fit3_new`** w zależności od liczby epok treningu. 

- **Oś X**: Reprezentuje liczbę epok treningowych.
- **Oś Y**: Pokazuje błąd klasyfikacji modelu. Niższy błąd oznacza lepszą jakość klasyfikacji.

### Interpretacja:
- Wartość błędu klasyfikacji **maleje** z każdą kolejną epoką, co wskazuje, że model stopniowo poprawia swoje prognozy.
- Po kilku epokach błąd stabilizuje się, co sugeruje, że model osiągnął optymalną jakość klasyfikacji na danych treningowych.

To oznacza, że model dobrze się uczy i nie przeucza się (brak wzrostu błędu po pewnym etapie).           

```{r}
# Wykres błędu klasyfikacji dla modelu 3
plot(dl_fit3_new, 
     timestep = "epochs", 
     metric = "classification_error")

```

Wykres przedstawia **błąd klasyfikacji** modelu w kroswalidacji w zależności od liczby epok. Oś X to liczba **epok treningowych**, a oś Y to **błąd klasyfikacji** modelu.

- **Niebieska linia (Training)**: Przedstawia błąd klasyfikacji na zbiorze treningowym w każdej epoce.
- **Pomarańczowa linia (Validation)**: Przedstawia błąd klasyfikacji na zbiorze walidacyjnym w każdej epoce.

### Interpretacja:
- **Błąd klasyfikacji na zbiorze treningowym** (niebieska linia) **maleje** z każdą epoką, co wskazuje, że model stopniowo poprawia swoje prognozy.
- **Błąd klasyfikacji na zbiorze walidacyjnym** (pomarańczowa linia) również maleje, ale w pewnym momencie może zacząć się **stabilizować** lub **wzrastać**, co może wskazywać na **przeuczenie modelu (overfitting)**. Model zaczyna się dobrze dopasowywać do danych treningowych, ale nie generalizuje dobrze na danych walidacyjnych.

```{r}
# Pobieramy modele z kroswalidacji dla modelu 3
cv_models_new <- sapply(dl_fit3_new@model$cross_validation_models, 
                        function(i) h2o.getModel(i$name))

# Wykres historii wyników dla pierwszego modelu w kroswalidacji
plot(cv_models_new[[1]], 
     timestep = "epochs", 
     metric = "classification_error")

```
## tutaj co się zmienia



**Definicja hiperparametrów dla modelu**:
   - **`activation_opt_new`**: Określenie dostępnych funkcji aktywacji, które będą używane w modelu (np. `Rectifier`, `Maxout`, `Tanh`).
   - **`l1_opt_new`** i **`l2_opt_new`**: Określają zakres wartości dla współczynników regularizacji L1 i L2, które są stosowane w modelu, aby zapobiegać przeuczeniu.
   - **`hyper_params_new`**: Tworzy listę z określonymi hiperparametrami, które będą używane podczas trenowania modelu, w tym funkcje aktywacji i współczynniki regularizacji.

```{r}
# Definiujemy hiperparametry do wyszukiwania
activation_opt_new <- c("Rectifier", "Maxout", "Tanh")
l1_opt_new <- c(0, 0.00001, 0.0001, 0.001, 0.01)
l2_opt_new <- c(0, 0.00001, 0.0001, 0.001, 0.01)

hyper_params_new <- list(activation = activation_opt_new, l1 = l1_opt_new, l2 = l2_opt_new)
search_criteria_new <- list(strategy = "RandomDiscrete", max_runtime_secs = 600)

```

**Podział danych na zestaw treningowy i walidacyjny**:
 - **`h2o.splitFrame()`**: Dzieli dane (zbiór `train_new`) na dwie części: zestaw treningowy (80% danych) i walidacyjny (20% danych). Jest to kluczowy krok w uczeniu maszynowym, aby ocenić wydajność modelu na nowych, niewidzianych danych.
   

```{r}
# Dzielimy dane na zestaw treningowy i walidacyjny
splits_new <- h2o.splitFrame(train_new, ratios = 0.8, seed = 1)

```

### Tworzenie siatki hiperparametrów dla modelu głębokiego uczenia

Funkcja **`h2o.grid()`** tworzy siatkę hiperparametrów, która testuje różne kombinacje ustawień modelu w celu znalezienia najlepszych parametrów. 

- **`x`** i **`y`**: Zmienna **`x`** to cechy, a **`y`** to zmienna celu, którą model będzie przewidywał.
- **`grid_id`**: Określa unikalny identyfikator dla siatki hiperparametrów.
- **`training_frame`** i **`validation_frame`**: Zbiory danych treningowych i walidacyjnych, na których model będzie trenowany i testowany.
- **`hidden`**: Liczba warstw ukrytych w sieci neuronowej (w tym przypadku 2 warstwy po 20 neuronów).
- **`hyper_params`**: Lista hiperparametrów, takich jak funkcje aktywacji, współczynniki regularizacji.
- **`search_criteria`**: Kryteria przeszukiwania przestrzeni hiperparametrów, np. metoda losowa.


```{r}
# Tworzymy siatkę hiperparametrów dla głębokiego uczenia
dl_grid_new <- h2o.grid("deeplearning", x = x_new, y = y_new,
                        grid_id = "dl_grid_new",
                        training_frame = splits_new[[1]],
                        validation_frame = splits_new[[2]],
                        seed = 1,
                        hidden = c(20, 20),
                        hyper_params = hyper_params_new,
                        search_criteria = search_criteria_new)

```

Pobieramy wyniki z **grid search** i sortujemy je według dokładności (accuracy) modelu, aby uzyskać najlepsze wyniki.

```{r}
# Sprawdzamy wyniki grid search dla najlepszych modeli
dl_gridperf_new <- h2o.getGrid(grid_id = "dl_grid_new", 
                               sort_by = "accuracy", 
                               decreasing = TRUE)
print(dl_gridperf_new)

```
```{r}
# Pobieramy najlepszy model z siatki
best_dl_model_id_new <- dl_gridperf_new@model_ids[[1]]
best_dl_new <- h2o.getModel(best_dl_model_id_new)

```
```{r}
# Oceniamy najlepszy model na zestawie testowym
best_dl_perf_new <- h2o.performance(model = best_dl_new, newdata = test_new)
h2o.mse(best_dl_perf_new)

```




### Generalized Linear Model (GLM)



Generalized Linear Model (GLM) to uogólnienie klasycznego modelu regresji liniowej, które pozwala modelować zależność między zmienną objaśnianą a zmiennymi objaśniającymi przy założeniu, że zmienna zależna pochodzi z rozkładu z rodziny wykładniczej (np. normalny, Poissona, Bernoulliego). Model GLM składa się z trzech elementów: funkcji łączącej (łączącej wartość oczekiwaną zmiennej zależnej z kombinacją liniową predyktorów), rozkładu prawdopodobieństwa oraz funkcji wariancji.


```{r}
data_file <- "./data/loan.csv"
if (!file.exists(data_file)) {
  data_file <- "https://raw.githubusercontent.com/ledell/LatinR-2019-h2o-tutorial/master/data/loan.csv"
} 
data <- h2o.importFile(data_file)
dim(data)
```
```{r}
nrows_subset <- 30000
data <- data[1:nrows_subset, ]
```
```{r}
data$bad_loan <- as.factor(data$bad_loan) 
h2o.levels(data$bad_loan) 
```
```{r}
h2o.describe(data)
```
```{r}
splits <- h2o.splitFrame(data = data, 
                         ratios = c(0.7, 0.15),  # partition data into 70%, 15%, 15% chunks
                         destination_frames = c("train", "valid", "test"), # frame ID (not required)
                         seed = 1)  # setting a seed will guarantee reproducibility
train <- splits[[1]]
valid <- splits[[2]]
test <- splits[[3]]

nrow(train)
nrow(valid)
nrow(test)
```
```{r}
y <- "bad_loan"
x <- setdiff(names(data), c(y, "int_rate"))
print(x)
```
```{r}
glm_fit1 <- h2o.glm(x = x, 
                    y = y, 
                    training_frame = train,
                    family = "binomial")
```
- Użycie funkcji h2o.glm() z pakietu H2O w R do stworzenia modelu GLM.  
- training_frame = train — zbiór danych treningowych.  
- family = "binomial" — model regresji logistycznej (klasyfikacja binarna).  
- Domyślna funkcja linku to `logit`, typowa dla regresji logistycznej.

```{r}
glm_fit1@model$model_summary
```
- Elastic Net łączy regularizację L1 i L2 z parametrem `alpha = 0.5`, a `lambda = 8.195E-5` określa siłę kary za złożoność modelu.  
- Model jest trenowany iteracyjnie, aż osiągnie optymalne dopasowanie.  
- Do modelu podano 82 predyktory.  
- Po regularizacji aktywnych zostało 54 predyktorów, co oznacza usunięcie mniej istotnych zmiennych.  
- Model zakończył dopasowanie po 4 iteracjach, co wskazuje na szybkie zbieżenie.  
- Uczenie odbywa się na zbiorze treningowym `train`.  
- Regularizacja pomaga zapobiegać przeuczeniu, poprawiając zdolność modelu do generalizacji.


```{r}
head(h2o.coef(glm_fit1))
```

- Intercept (-1.53): wartość bazowa modelu, gdy stan to Alaska (stan referencyjny).
- addr_state.AK = 0: Alaska jest stanem odniesienia, brak dodatkowego wpływu.
- addr_state.AL = -0.28: Wartość zmiennej jest niższa o 0.28 w Alabamie w porównaniu do Alaski.
- addr_state.AR = 0: Arkansas ma taki sam wpływ jak Alaska (brak różnicy).
- addr_state.AZ = -0.12: Arizona obniża wartość zmiennej o 0.12 względem Alaski.
- addr_state.CA = 0.17: Kalifornia podnosi wartość zmiennej o 0.17 w stosunku do Alaski.

Podsumowując, względem Alaski, Kalifornia ma pozytywny wpływ na zmienną, Alabama i Arizona – negatywny, a Arkansas jest neutralny.

```{r}
h2o.varimp_plot(glm_fit1)
```

Najważniejsze zmienne w modelu to: długość okresu pożyczki (term.36 months, term.60 months), cel pożyczki (np. small_business, car, wedding, credit_card) oraz stan zamieszkania (addr_state.NV, addr_state.DC itd.). Te cechy mają największy wpływ na przewidywania modelu.

```{r}
glm_perf1 <- h2o.performance(model = glm_fit1,
                             newdata = test)
print(glm_perf1)
```
## Interpretacja wyników modelu GLM z pakietu H2O

### Metryki modelu

- **MSE (Mean Squared Error):** 0.1241 — średni kwadrat błędu; niższa wartość oznacza lepsze dopasowanie.
- **RMSE (Root Mean Squared Error):** 0.3523 — pierwiastek z MSE, wyrażony w jednostkach oryginalnej zmiennej.
- **LogLoss:** 0.4050 — miara błędu w klasyfikacji binarnej; im niższa, tym lepiej.
- **Mean Per-Class Error:** 0.3360 — średni błąd klasyfikacji per klasa.
- **AUC (Area Under Curve):** 0.7154 — ocena jakości klasyfikatora; wartość bliższa 1 oznacza lepszy model.
- **AUCPR:** 0.3492 — pole pod krzywą precyzja-czułość.
- **Gini:** 0.4290 — alternatywna miara jakości (Gini = 2 * AUC - 1).
- **R²:** 0.0942 — niska wartość wskazuje na słabe wyjaśnienie wariancji przez model.
- **Residual Deviance:** 3638.326 — suma odchyleń resztowych; im niższa, tym lepiej.
- **AIC:** 3748.326 — kryterium informacyjne dla modelu; niższe wartości preferowane.

---

### Macierz konfuzji

- Model dobrze rozpoznaje klasę 0 (77.6% poprawnych przewidywań).
- Klasa 1 jest trudniejsza do klasyfikacji (44.87% błędów).
- Całkowity błąd klasyfikacji to 26%.

---

threshold - Wartość progu (threshold), przy którym osiągnięto podaną wartość metryki. W klasyfikacji to granica decydująca o przypisaniu klasy.
value - Zmierzona wartość metryki przy danym progu, pokazująca jakość modelu według tej metryki.
idx - Indeks lub identyfikator próbki/danych, przy którym osiągnięto tę wartość metryki.

---

### Podsumowanie

- Model ma umiarkowaną jakość (AUC 0.7154), ale wysoki błąd na klasie 1.
- Model lepiej rozpoznaje klasę 0 niż 1.
- W zależności od celu można dostosować próg decyzyjny, by maksymalizować różne metryki.
- Zalecane dalsze dostosowanie modelu lub analiza, jeśli klasa 1 jest kluczowa.


```{r}
h2o.auc(glm_perf1)
```
```{r}
preds <- predict(glm_fit1, newdata = test)
head(preds)
```

- **predict**: Przewidywana klasa (etykieta) dla danej obserwacji, wynikająca z modelu.
- **p0**: Prawdopodobieństwo, że obserwacja należy do klasy 0.
- **p1**: Prawdopodobieństwo, że obserwacja należy do klasy 1.


```{r}
glm_fit2 <- h2o.glm(x = x, 
                    y = y, 
                    training_frame = train,
                    family = "binomial",
                    nfolds = 5,
                    seed = 1)
```

- nfolds: liczba podziałów danych w walidacji krzyżowej (cross-validation).  
Dane treningowe są dzielone na *nfolds* części, a model jest trenowany i testowany *nfolds* razy, za każdym razem na innych podzbiorach.  
Pozwala to na rzetelną ocenę jakości modelu i jego zdolności do generalizacji.

```{r}
h2o.auc(glm_fit2, xval = TRUE)
```
