# Bank Scoring ML Project

Модель для оценки риска дефолта клиентов на основе данных о кошельках.

## 🚀 Запуск

1. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   
# scoring_ml-project

md

# Bank Scoring ML Project

Модель машинного обучения для оценки вероятности дефолта клиентов банка. Проект включает EDA, обучение модели, оценку качества, интерпретацию признаков и подготовку к деплою.

## 📊 Результаты

- **Train ROC AUC**: 0.9479  
- **Validation ROC AUC**: 0.9231  
- **Test ROC AUC**: 0.9231  
- **Cross-Validation ROC AUC**: 0.9215 ± 0.0032  
- **Оптимальный порог (по F1)**: 0.3843  

### Classification Report (Test, порог = 0.3843)

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.87      | 0.89   | 0.88     | 69,062  |
| 1     | 0.81      | 0.79   | 0.80     | 41,679  |
| **Accuracy** | — | — | **0.85** | **110,741** |

Модель показывает **стабильное качество** на всех выборках, без переобучения.

## 🛠️ Как запустить

### Требования
- Python 3.9+
- pip

### Установка зависимостей
```bash
pip install -r requirements.txt
Обучение модели

bash
python src/train_pipeline.py
Оценка и генерация отчётов

bash
python src/evaluate.py
Запуск API (Flask)

bash
python app/app.py
API будет доступен на http://localhost:5000/predict.

⚠️ Данные (data/dataset.parquet) должны находиться в папке data/ локально.
Файл данных не включён в репозиторий из-за ограничений GitHub (>100 МБ). 

📁 Структура проекта

scoring_ml-project/
├── app/               # Flask API
├── src/               # Обучение и оценка модели
├── data/              # Локальные данные (не в Git!)
├── models/            # Сохранённые модели (не в Git!)
├── results/           # Отчёты, CSV, метрики (не в Git!)
├── requirements.txt   # Зависимости
└── README.md
🧠 Используемые технологии
LightGBM (устойчив к выбросам, быстрый инференс)
SHAP — интерпретация признаков
Optuna — подбор гиперпараметров (если использовал)
Flask + Docker — подготовка к деплою
📌 Примечание
Проект разработан в рамках сотрудничества с банком.
---
 
git add README.md .gitignore
git commit -m "Add project documentation and ignore large files"
git push
