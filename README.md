# HR Analytics: Employee Satisfaction & Turnover Prediction

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.4.2-013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-3.0.1-150458?logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8.0-11557c?logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13.2-4c72b0?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-F7931E?logo=scikit-learn&logoColor=white)
![Phik](https://img.shields.io/badge/Phik-0.12.5-6f42c1?logo=python&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-0.46.0-ff69b4?logo=python&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-0.14.2-1C3F5C?logo=python&logoColor=white)

## Описание проекта

Проект разработан для компании «Работа с заботой» в рамках учебного курса по Data Science.  
Цель — создать модели машинного обучения, которые помогут HR-отделу прогнозировать два ключевых показателя:

1. **Уровень удовлетворённости сотрудника** (регрессионная задача).  
2. **Вероятность увольнения сотрудника** (задача бинарной классификации).

Решение позволяет компании своевременно выявлять риски, снижать текучесть кадров и минимизировать финансовые потери, связанные с внезапным уходом ценных сотрудников.

## Данные

Все необходимые файлы находятся в папке `data/` репозитория:

- `train_job_satisfaction_rate.csv` – обучающая выборка для задачи удовлетворённости (с целевой переменной).
- `train_quit.csv` – обучающая выборка для задачи увольнения.
- `test_features.csv` – признаки тестовой выборки (общие для обеих задач).
- `test_target_job_satisfaction_rate.csv` – целевая переменная удовлетворённости для теста.
- `test_target_quit.csv` – целевая переменная увольнения для теста.

**Описание признаков:**

- `id` – уникальный идентификатор сотрудника;
- `dept` – отдел;
- `level` – уровень должности (junior, middle, senior);
- `workload` – загруженность (low, medium, high);
- `employment_years` – стаж работы в компании (в годах);
- `last_year_promo` – было ли повышение за последний год;
- `last_year_violations` – были ли нарушения трудового договора;
- `supervisor_evaluation` – оценка руководителя (от 1 до 5);
- `salary` – ежемесячная зарплата;
- `job_satisfaction_rate` – уровень удовлетворённости (цель 1);
- `quit` – факт увольнения (цель 2).

## Этапы работы

1. **Загрузка и первичный анализ**  
   - Единая функция для импорта и вывода информации о данных.
   - Проверка пропусков, дубликатов, типов данных.
   - Установка `id` в качестве индекса.

2. **Разведочный анализ (EDA)**  
   - Визуализация распределений категориальных и числовых признаков.
   - Корреляционный анализ с использованием `phik` для смешанных типов данных.
   - Анализ мультиколлинеарности (VIF).

3. **Предобработка данных и пайплайны**  
   - Для каждой задачи сформированы матрицы признаков и целевые векторы.
   - Созданы `ColumnTransformer` и `Pipeline` с обработкой пропусков, кодированием категорий (OneHotEncoder, OrdinalEncoder) и масштабированием числовых признаков.
   - OrdinalEncoder настроен с явным указанием категорий и обработкой неизвестных значений.

4. **Моделирование (удовлетворённость)**  
   - Метрика: SMAPE (симметричная средняя абсолютная процентная ошибка).
   - Модели: `LinearRegression`, `DecisionTreeRegressor`.
   - Подбор гиперпараметров через `RandomizedSearchCV` (5-кратная кросс-валидация).
   - **Лучшая модель:** `DecisionTreeRegressor` с параметрами `max_depth=17`, `min_samples_split=12`, `max_features=12`.
   - SMAPE на тестовой выборке: **13.74%**.
   - Анализ остатков подтвердил нормальность распределения и гомоскедастичность.

5. **Моделирование (увольнение)**  
   - Метрика: ROC-AUC.
   - Модели: `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `KNeighborsClassifier`.
   - Подбор гиперпараметров с помощью `RandomizedSearchCV`.
   - **Лучшая модель:** `RandomForestClassifier` (ROC-AUC = **0.87** на тесте).

6. **Интерпретация моделей**  
   - Анализ важности признаков для дерева решений.
   - Использование SHAP для объяснения предсказаний классификатора.
   - Наибольшее влияние на целевую переменную оказывают `supervisor_evaluation` и `last_year_violations`.

## Используемые технологии

- **Python** — основной язык программирования.
- **Pandas, NumPy** — обработка и анализ данных.
- **Matplotlib, Seaborn** — визуализация.
- **Phik** — корреляционный анализ для смешанных типов данных.
- **Scikit-learn** — предобработка, пайплайны, модели, подбор гиперпараметров, метрики.
- **SHAP** — интерпретация моделей.
- **statsmodels** — проверка мультиколлинеарности (VIF).

## Результаты

| Задача              | Модель                | Метрика        | Значение |
|---------------------|-----------------------|----------------|----------|
| Удовлетворённость   | DecisionTreeRegressor | SMAPE (test)   | 13.74%   |
| Увольнение          | RandomForestClassifier| ROC-AUC (test) | 0.87     |

Ключевые факторы, влияющие на удовлетворённость и увольнение: **оценка руководителя** и **наличие нарушений**. Зарплата и стаж имеют умеренное влияние, отдел и уровень должности — минимальное.

## Структура репозитория

```
hr_analytics_care_prediction/
├── data/                          # исходные файлы
│   ├── test_features.csv
│   ├── test_target_job_satisfaction_rate.csv
│   ├── test_target_quit.csv
│   ├── train_job_satisfaction_rate.csv
│   └── train_quit.csv
├── notebooks/                     # Jupyter ноутбук с полным анализом
│   └── hr_analytics_care_prediction.ipynb
├── README.md                      # этот файл
└── requirements.txt               # зависимости
```

## Запуск проекта

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/arseniybartenev/hr_analytics_care_prediction.git
   cd hr_analytics_care_prediction
   ```

2. Установите зависимости (рекомендуется использовать виртуальное окружение):
   ```bash
   pip install -r requirements.txt
   ```

3. Откройте Jupyter Notebook и последовательно выполните все ячейки:
   ```bash
   jupyter notebook notebooks/hr_analytics_care_prediction.ipynb
   ```

## Контакты

Автор: Arseniy Bartenev  
Email: arseniybartenev@gmail.com
