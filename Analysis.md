# Анализ репозитория SMILES-HALLUCINATION-DETECTION

## 📌 Кратко о задаче

**Цель:** Создать классификатор (probe), который по внутренним представлениям (hidden states) языковой модели определяет, является ли ответ модели **галлюцинацией** (выдумкой) или **правдивым**.

**Модель:** Qwen2.5-0.5B — небольшой трансформер (24 слоя, размер скрытого слоя 896).

**Метрика:** Accuracy (точность) на тестовом наборе.

---

## 📁 Структура репозитория

```
SMILES-HALLUCINATION-DETECTION/
├── data/
│   ├── dataset.csv      # Обучающие данные (689 примеров): prompt, response, label
│   └── test.csv         # Тестовый набор без меток (для предсказания)
│
├── solution.py          # Главный скрипт — запускает весь пайплайн
│
├── --- ФАЙЛЫ ДЛЯ РЕДАКТИРОВАНИЯ --- 
├── aggregation.py       # Агрегация hidden states → feature vector
├── probe.py             # Классификатор HallucinationProbe
├── splitting.py         # Стратегия разбиения на train/val/test
│
├── --- ИНФРАСТРУКТУРА (НЕ МЕНЯТЬ) ---
├── model.py             # Загрузка модели Qwen2.5-0.5B
├── evaluate.py          # Оценка метрик, сохранение результатов
│
├── requirements.txt     # Зависимости Python
└── README.md            # Документация
```

---

## 🔍 Как работает пайплайн (solution.py)

1. **Загрузка данных:** Чтение `dataset.csv`, объединение `prompt + response` в один текст.

2. **Экстракция hidden states:** 
   - Текст токенизируется и подаётся в модель.
   - Модель возвращает hidden states для каждого слоя (25 тензоров: эмбеддинги + 24 слоя).
   - Форма: `(n_layers, seq_len, hidden_dim)` = `(25, до 512, 896)`.

3. **Агрегация (aggregation.py):** 
   - Из 3D тензора получается 1D вектор признаков.
   - По умолчанию: берётся последний реальный токен последнего слоя.

4. **Обучение классификатора (probe.py):**
   - На feature vectors обучается бинарный классификатор (MLP).
   - Предсказывает: 0 = правдиво, 1 = галлюцинация.

5. **Оценка (evaluate.py):**
   - Считаются Accuracy, F1, AUROC на train/val/test.
   - Результаты сохраняются в `results.json`.

6. **Предсказание на тесте:**
   - Для `test.csv` извлекаются признаки.
   - Обученный probe делает предсказания.
   - Результат сохраняется в `predictions.csv`.

---

## 🛠️ Что нужно реализовать (3 файла)

### 1. `aggregation.py` — Агрегация признаков

**Функция `aggregate()`** — превращает тензор скрытых состояний в вектор признаков.

**Текущая реализация (baseline):**
```python
# Берёт последний слой, последний реальный токен
layer = hidden_states[-1]  # (seq_len, hidden_dim)
last_pos = int(attention_mask.nonzero()[-1].item())
feature = layer[last_pos]  # (hidden_dim,)
```

**Варианты улучшения:**

| Стратегия | Описание |
|-----------|----------|
| **Mean pooling** | Среднее по всем реальным токенам |
| **Max pooling** | Максимум по всем токенам |
| **CLS-like** | Первый токен (как в BERT) |
| **Multi-layer** | Конкатенация нескольких слоёв (например, 20, 22, 24) |
| **Weighted sum** | Взвешенная сумма токенов по attention |

**Пример — mean pooling:**
```python
def aggregate(hidden_states, attention_mask):
    layer = hidden_states[-1]  # последний слой
    real_tokens = layer[attention_mask == 1]  # только реальные токены
    return real_tokens.mean(dim=0)  # среднее по токенам
```

**Функция `extract_geometric_features()`** — дополнительные hand-crafted признаки.

**Идеи для геометрических признаков:**
- Норма активаций каждого слоя (L2 norm)
- Косинусное сходство между слоями (representation drift)
- Дисперсия активаций по токенам
- Длина последовательности

---

### 2. `probe.py` — Классификатор

**Класс `HallucinationProbe`** — бинарный классификатор на PyTorch.

**Текущая реализация (baseline):**
```python
nn.Sequential(
    nn.Linear(input_dim, 256),
    nn.ReLU(),
    nn.Linear(256, 1),  # логит
)
```

**Варианты улучшения:**

| Модификация | Описание |
|-------------|----------|
| **Более глубокая сеть** | Добавить слои: 896 → 512 → 256 → 128 → 1 |
| **Dropout** | Добавить регуляризацию |
| **BatchNorm** | Нормализация батчей |
| **Разные активации** | GELU, SiLU вместо ReLU |
| **Dimensionality reduction** | PCA/Autoencoder перед классификатором |

**Пример — сеть с Dropout:**
```python
def _build_network(self, input_dim):
    self._net = nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(256, 1),
    )
```

**Варианты обучения:**
- Увеличить количество эпох (сейчас 200)
- Снизить learning rate с расписанием (lr_scheduler)
- Использовать более продвинутые оптимизаторы (AdamW)

---

### 3. `splitting.py` — Разбиение данных

**Функция `split_data()`** — делит данные на train/val/test.

**Текущая реализация (baseline):**
- Один стратифицированный сплит: 70% train, 15% val, 15% test

**Варианты улучшения:**

| Стратегия | Описание |
|-----------|----------|
| **K-Fold Cross-Validation** | 5 или 10 фолдов для стабильной оценки |
| **Grouped split** | Группировка по похожим prompt'ам |
| **Time-based split** | Если есть временная метка |

**Пример — 5-fold CV:**
```python
from sklearn.model_selection import KFold

def split_data(y, df, n_folds=5, random_state=42):
    kf = KFold(n_splits=n_folds, shuffle=True, stratify=y, random_state=random_state)
    splits = []
    indices = np.arange(len(y))
    
    for train_val_idx, test_idx in kf.split(indices, y):
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=0.15, 
            stratify=y[train_val_idx], random_state=random_state
        )
        splits.append((train_idx, val_idx, test_idx))
    
    return splits
```

---

## 📋 План решения

### Этап 1: Baseline (быстрый старт)
1. Запустить `python solution.py` без изменений.
2. Зафиксировать метрики baseline.

### Этап 2: Улучшение агрегации
1. Попробовать mean pooling вместо last token.
2. Попробовать конкатенацию нескольких последних слоёв.
3. Добавить геометрические признаки (нормы, дисперсии).

### Этап 3: Улучшение классификатора
1. Добавить Dropout.
2. Поэкспериментировать с архитектурой (глубина, ширина).
3. Попробовать lr_scheduler.

### Этап 4: Улучшение валидации
1. Реализовать k-fold cross-validation.
2. Это даст более надёжную оценку.

### Этап 5: Финальная сборка
1. Выбрать лучшие компоненты.
2. Запустить финальный прогон.
3. Сохранить `predictions.csv` и `results.json`.

---

## 💡 Ключевые идеи для экспериментов

### Агрегация (aggregation.py)

```python
# Идея 1: Multi-layer concatenation
def aggregate(hidden_states, attention_mask):
    layers_to_use = [-1, -2, -3]  # последние 3 слоя
    features = []
    for layer_idx in layers_to_use:
        layer = hidden_states[layer_idx]
        real_tokens = layer[attention_mask == 1]
        features.append(real_tokens.mean(dim=0))
    return torch.cat(features)  # размер: 3 * 896 = 2688
```

```python
# Идея 2: Геометрические признаки
def extract_geometric_features(hidden_states, attention_mask):
    norms = []
    for layer in hidden_states[1:]:  # пропустить эмбеддинги
        real_tokens = layer[attention_mask == 1]
        norm = real_tokens.norm(dim=1).mean()
        norms.append(norm)
    
    # Добавляем дисперсию по токенам
    variances = []
    for layer in hidden_states[1:]:
        real_tokens = layer[attention_mask == 1]
        var = real_tokens.var(dim=0).mean()
        variances.append(var)
    
    return torch.cat([torch.tensor(norms), torch.tensor(variances)])
```

### Классификатор (probe.py)

```python
# Идея: Сеть с residual-like структурой
def _build_network(self, input_dim):
    self._net = nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.BatchNorm1d(512),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(128, 1),
    )
```

---

## 📊 Результаты Baseline (из results.json)

```json
{
  "n_samples": 689,
  "n_folds": 1,
  "feature_dim": 896,
  "extract_time_s": 136.05,
  
  "avg_train_accuracy": 0.9605,    // 96.05%
  "avg_val_accuracy": 0.7404,      // 74.04%
  "avg_test_accuracy": 0.7404,     // 74.04%
  
  "avg_train_auroc": 0.9998,       // почти идеально на train
  "avg_val_auroc": 0.6589,         // ~66% на валидации
  "avg_test_auroc": 0.7479,        // ~75% на тесте
  
  "baseline_accuracy": 0.7019      // просто majority class
}
```

### 🔍 Анализ baseline

| Метрика | Train | Val | Test | Вывод |
|---------|-------|-----|-----|-------|
| **Accuracy** | 96.0% | 74.0% | 74.0% | Сильное переобучение |
| **F1** | 97.3% | 83.2% | 82.6% | На валидации лучше |
| **AUROC** | 99.98% | 65.9% | 74.8% | **Проблема на валидации** |

### ❗ Ключевые проблемы

1. **Сильное переобучение (overfitting)**
   - Разрыв train vs val accuracy: 96% → 74% = **22 процентных пункта**
   - Train AUROC почти 100%, но val AUROC всего 66%
   - Модель заучивает примеры, а не учится обобщать

2. **Низкий AUROC на валидации (0.66)**
   - Это близко к случайному угадыванию (0.5)
   - Модель плохо ранжирует примеры по уверенности
   - F1 выше (83%), значит модель умеет предсказывать, но не уверенно

3. **Feature dimension = 896**
   - Используется только последний токен последнего слоя
   - Теряется информация из других слоёв и токенов
   - Нет геометрических признаков

### ✅ Что уже работает

- Test accuracy (74%) выше baseline (70%) — модель учится чему-то
- F1 на тесте (82.6%) — приличное значение
- Разрыв между val и test отсутствует (оба 74%) — случайность сплита не влияет

---

## 📈 Потенциал улучшений

| Компонент | Текущее | Потенциал | Ожидаемый прирост |
|-----------|---------|-----------|-------------------|
| **Агрегация** | Last token (896) | Multi-layer + pooling | +3-7% accuracy |
| **Геометрические признаки** | Нет | Norms, variance, drift | +2-4% accuracy |
| **Dropout** | Нет | 0.2-0.4 | -2-5% overfitting |
| **Классификатор** | 1 hidden layer | 2-3 слоя + BatchNorm | +1-3% accuracy |
| **K-Fold CV** | 1 fold | 5-10 folds | Стабильная оценка |

**Ожидаемая итоговая accuracy:** 78-82% (с уменьшением переобучения)

---

## 🚀 Быстрый старт

```bash
# 1. Установка зависимостей
pip install -r requirements.txt

# 2. Запуск baseline
python solution.py

# 3. Просмотр результатов
cat results.json
cat predictions.csv
```

---

## 📝 Что нужно для отправки

1. **GitHub репозиторий** с вашим кодом.
2. **results.json** — сгенерированный solution.py.
3. **predictions.csv** — предсказания на test.csv.
4. **SOLUTION.md** — отчёт с описанием решения.
