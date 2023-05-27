# <YOUR_IMPORTS>
import os
import dill
import pandas as pd
import json



def predict():
    def load_model_from_folder(folder_path):
        # Получаем список файлов в папке
        files = os.listdir(folder_path)

        # Поиск файла модели с расширением .pkl
        for file_name in files:
            if file_name.endswith('.pkl'):
                # Полный путь к файлу модели
                model_path = os.path.join(folder_path, file_name)
                # Загрузка модели
                with open(model_path, 'rb') as f:
                    model = dill.load(f)
                return model

        # Если не найдено ни одного файла модели, пишем исключение
        raise FileNotFoundError("No model file found in the specified folder.")

    # Загружаем модель:
    folder_path = 'data/models'
    model = load_model_from_folder(folder_path)

    # Получаем список файлов из папки test
    test_files = os.listdir('data/test')

    # Список для хранения предсказаний
    predictions = []

    # Предсказание для каждого файла в папке test
    for file_name in test_files:
        # Полный путь к файлу
        file_path = os.path.join('data/test', file_name)

        # Загрузка данных для предсказания
        with open(file_path, 'rb') as f:
            data = json.load(f)
            input_data = pd.DataFrame(data, index=[0])

        # Выполнение предсказания
        prediction = model.predict(input_data)

        # Добавление в список нового предсказания
        predictions.append(prediction)

    # Создание DataFrame из списка предсказаний
    predictions_df = pd.DataFrame(predictions)

    # Сохранение предсказаний в csv-файл
    predictions_df.to_csv('data/predictions/predictions.csv', index=False)

if __name__ == '__main__':
    predict()

