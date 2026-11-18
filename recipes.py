#!/usr/bin/env python3
"""
recipes.py

Модуль с классами и методами для приложения "Питание и рацион".
Используется основным скриптом nutritionist.py.

Артефакты, которые ожидаются рядом с этим файлом (готовятся в recipes.ipynb):
    - model.pkl              -> обученная модель предсказания рейтинга/класса
    - model_columns.pkl      -> список ингредиентных колонок, на которых
                                 обучалась модель (порядок важен!)
    - nutrition_facts.csv    -> DataFrame: ингредиент -> % суточной нормы по нутриентам
    - recipes_similarity.csv -> DataFrame: рецепты с ингредиентами, рейтингом,
                                 URL и (опционально) типом приёма пищи
"""

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model.pkl"
MODEL_COLUMNS_PATH = BASE_DIR / "model_columns.pkl"
NUTRITION_FACTS_PATH = BASE_DIR / "nutrition_facts.csv"
RECIPES_SIMILARITY_PATH = BASE_DIR / "recipes_similarity.csv"

RATING_CLASSES = ("bad", "so-so", "great")


def normalize_ingredient(name: str) -> str:
    """Приводит название ингредиента к единому виду для сравнения/поиска."""
    name = name.strip().lower()
    name = re.sub(r"[^a-zа-яё0-9\s-]", "", name)
    return name


class RatingPredictor:
    """
    Обёртка над обученной моделью (сохранённой в recipes.ipynb).
    Модель предсказывает класс рейтинга блюда ('bad' / 'so-so' / 'great')
    по бинарному вектору наличия ингредиентов.
    """

    def __init__(self, model_path=MODEL_PATH, columns_path=MODEL_COLUMNS_PATH):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(columns_path, "rb") as f:
            # список ингредиентных фичей в том порядке, на котором обучалась модель
            self.feature_columns = pickle.load(f)
        self._known_normalized = {normalize_ingredient(c) for c in self.feature_columns}

    def known_ingredients(self):
        """Словарь известных модели ингредиентов (нормализованные имена)."""
        return self._known_normalized

    def missing_ingredients(self, ingredients):
        """Возвращает список ингредиентов из query, которых нет в нашей базе."""
        return [
            ing for ing in ingredients
            if normalize_ingredient(ing) not in self._known_normalized
        ]

    def _vectorize(self, ingredients):
        normalized = {normalize_ingredient(i) for i in ingredients}
        values = [
            1 if normalize_ingredient(col) in normalized else 0
            for col in self.feature_columns
        ]
        # DataFrame (not a bare ndarray) so sklearn sees the same feature
        # names it was trained with and doesn't warn/misalign columns.
        return pd.DataFrame([values], columns=self.feature_columns)

    def predict_class(self, ingredients):
        """
        Возвращает один из RATING_CLASSES.
        Если модель - регрессор, переводим числовой рейтинг в класс,
        округляя и группируя по правилу bad/so-so/great.
        """
        vector = self._vectorize(ingredients)
        prediction = self.model.predict(vector)[0]

        # если модель классификационная и уже возвращает строку/класс
        if isinstance(prediction, str):
            return prediction

        # если регрессор - переводим число рейтинга (0..5) в класс
        rounded = int(round(prediction))
        rounded = max(0, min(5, rounded))
        if rounded <= 1:
            return "bad"
        elif rounded <= 3:
            return "so-so"
        else:
            return "great"


class NutritionFacts:
    """
    Работа с таблицей нутриентов по ингредиентам (nutrition_facts.csv),
    подготовленной через USDA FoodData Central API в recipes.ipynb.
    """

    def __init__(self, csv_path=NUTRITION_FACTS_PATH):
        self.df = pd.read_csv(csv_path)
        self.df["ingredient_norm"] = self.df["ingredient"].apply(normalize_ingredient)

    def get_facts_for_ingredient(self, ingredient):
        """Возвращает dict {нутриент: % суточной нормы} для одного ингредиента."""
        norm = normalize_ingredient(ingredient)
        row = self.df[self.df["ingredient_norm"] == norm]
        if row.empty:
            return {}
        row = row.iloc[0]
        nutrients = {}
        for col in self.df.columns:
            if col in ("ingredient", "ingredient_norm"):
                continue
            value = row[col]
            if pd.notna(value):
                nutrients[col] = value
        return nutrients

    def get_facts_for_ingredients(self, ingredients):
        """Возвращает dict {ингредиент: {нутриент: % от нормы}} для списка ингредиентов."""
        return {ing: self.get_facts_for_ingredient(ing) for ing in ingredients}

    def total_daily_value(self, ingredients):
        """
        Суммарные % суточной нормы по всем нутриентам для списка ингредиентов
        (используется, например, для сборки меню на день в бонусной части).
        """
        totals = {}
        for ing in ingredients:
            for nutrient, value in self.get_facts_for_ingredient(ing).items():
                totals[nutrient] = totals.get(nutrient, 0) + value
        return totals


class SimilarRecipesFinder:
    """
    Поиск наиболее похожих рецептов по списку ингредиентов
    (таблица recipes_similarity.csv, подготовленная в recipes.ipynb).
    Ожидаемые колонки: title, rating, url, ingredients (строка через запятую),
    и опционально meal_type (breakfast/lunch/dinner) для бонусной части.
    """

    def __init__(self, csv_path=RECIPES_SIMILARITY_PATH):
        self.df = pd.read_csv(csv_path)
        self.df["ingredients_set"] = self.df["ingredients"].apply(
            lambda s: {normalize_ingredient(i) for i in str(s).split(",") if i.strip()}
        )

    @staticmethod
    def _jaccard(set_a, set_b):
        if not set_a and not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    def find_top_similar(self, ingredients, top_n=3, meal_type=None, min_similarity=0.3):
        """
        Возвращает top_n рецептов (DataFrame-строки) с наибольшим
        коэффициентом Жаккара по набору ингредиентов.
        Если указан meal_type - ищет только среди рецептов этого типа
        (нужно для бонусной части: завтрак/обед/ужин).

        Если лучшее совпадение слабее min_similarity (по умолчанию 0.3),
        считаем, что действительно похожих рецептов нет, и возвращаем
        пустой DataFrame - так и должно быть для случайного/бессвязного
        набора из многих ингредиентов сразу.
        """
        def fix_encoding(title):
            if not isinstance(title, str):
                return title
            try:
                return title.encode('latin-1').decode('utf-8')
            except (UnicodeEncodeError, UnicodeDecodeError):
                return title

        query_set = {normalize_ingredient(i) for i in ingredients}
        subset = self.df
        if meal_type is not None and "meal_type" in self.df.columns:
            subset = subset[subset["meal_type"] == meal_type]

        scores = subset["ingredients_set"].apply(lambda s: self._jaccard(query_set, s))
        ranked = subset.assign(similarity=scores).sort_values(
            "similarity", ascending=False
        )
        if ranked.empty or ranked.iloc[0]["similarity"] < min_similarity:
            return ranked.head(0)
        
        top_df = ranked.head(top_n).copy()
        if "title" in top_df.columns:
            top_df["title"] = top_df["title"].apply(fix_encoding)

        return top_df


class Nutritionist:
    """
    Главный класс, объединяющий предсказание рейтинга, нутриенты
    и поиск похожих рецептов. Именно его использует nutritionist.py.
    """

    def __init__(self):
        self.predictor = RatingPredictor()
        self.nutrition_facts = NutritionFacts()
        self.recipe_finder = SimilarRecipesFinder()

    def missing_ingredients(self, ingredients):
        """Ингредиенты, которых нет в нашей базе (не встречались в датасете)."""
        return self.predictor.missing_ingredients(ingredients)

    def forecast(self, ingredients):
        """I. Прогноз: класс рейтинга блюда."""
        return self.predictor.predict_class(ingredients)

    def nutrition_report(self, ingredients):
        """II. Нутриенты по каждому ингредиенту."""
        return self.nutrition_facts.get_facts_for_ingredients(ingredients)

    def similar_recipes(self, ingredients, top_n=3):
        """III. Топ-N похожих рецептов."""
        return self.recipe_finder.find_top_similar(ingredients, top_n=top_n)

    def daily_menu(self, target_daily_value=90):
        """
        Бонус: генерирует меню на день (breakfast/lunch/dinner),
        случайно перебирая рецепты каждого типа и выбирая комбинацию,
        которая покрывает большую часть суточной нормы, не превышая её,
        и имеет максимальный суммарный рейтинг.
        """
        import itertools
        import random

        df = self.recipe_finder.df
        if "meal_type" not in df.columns:
            raise ValueError(
                "В recipes_similarity.csv нет колонки meal_type - "
                "она нужна для бонусной части (генерации меню)."
            )

        best_combo = None
        best_score = -1

        candidates = {
            meal: df[df["meal_type"] == meal].sample(
                min(15, len(df[df["meal_type"] == meal]))
            )
            for meal in ("breakfast", "lunch", "dinner")
        }

        for _ in range(500):
            picks = {
                meal: candidates[meal].sample(1).iloc[0]
                for meal in ("breakfast", "lunch", "dinner")
                if len(candidates[meal]) > 0
            }
            if len(picks) < 3:
                continue

            all_ingredients = sum(
                (list(row["ingredients_set"]) for row in picks.values()), []
            )
            totals = self.nutrition_facts.total_daily_value(all_ingredients)
            if any(v > target_daily_value for v in totals.values()):
                continue

            total_rating = sum(row["rating"] for row in picks.values())
            coverage = sum(totals.values())
            score = total_rating + coverage / 1000  # рейтинг важнее, покрытие - тай-брейкер

            if score > best_score:
                best_score = score
                best_combo = picks

        return best_combo


if __name__ == "__main__":
    # Небольшая самопроверка модуля (не заменяет полноценные тесты)
    print("recipes.py: модуль с классами Nutritionist, RatingPredictor, "
          "NutritionFacts, SimilarRecipesFinder загружен успешно.")
