#!/usr/bin/env python3
"""
nutritionist.py

Основная программа. Принимает список ингредиентов через аргументы
командной строки и печатает:
    I.   прогноз рейтинга блюда (bad / so-so / great)
    II.  нутриенты по каждому ингредиенту (в % от суточной нормы)
    III. топ-3 похожих рецепта с рейтингом и URL

Пример запуска:
    $ ./nutritionist.py milk, honey, jam
"""

import sys

from recipes import Nutritionist


def parse_ingredients(argv):
    """Собирает список ингредиентов из аргументов командной строки.
    Работает и для `milk, honey, jam` (одна строка с запятыми),
    и для `milk honey jam` (раздельные аргументы)."""
    raw = " ".join(argv)
    if "," in raw:
        items = raw.split(",")
    else:
        items = raw.split()
    return [item.strip() for item in items if item.strip()]


def print_forecast(rating_class):
    print("I. OUR FORECAST")
    messages = {
        "bad": (
            "You might find it tasty, but in our opinion, it is a bad idea to have a\n"
            "dish with that list of ingredients."
        ),
        "so-so": (
            "It might turn out just so-so - not a masterpiece, but not a disaster\n"
            "either. Proceed if you're feeling adventurous."
        ),
        "great": (
            "Great choice! This combination of ingredients has strong potential\n"
            "for a genuinely delicious dish."
        ),
    }
    print(messages.get(rating_class, rating_class))


def print_nutrition_report(report):
    print("II. NUTRITION FACTS")
    for ingredient, nutrients in report.items():
        print(ingredient.title())
        if not nutrients:
            print("  (no data found)")
            continue
        for nutrient, value in nutrients.items():
            print(f"{nutrient} - {value:.0f}% of Daily Value")


def print_similar_recipes(recipes_df):
    print("III. TOP-3 SIMILAR RECIPES:")
    if recipes_df.empty:
        print("No similar recipes were found for this combination of ingredients.")
        return
    for _, row in recipes_df.iterrows():
        print(f"- {row['title']}, rating: {row['rating']}, URL:")
        print(f"{row['url']}")


def main():
    ingredients = parse_ingredients(sys.argv[1:])
    if not ingredients:
        print("Usage: ./nutritionist.py ingredient1, ingredient2, ...")
        sys.exit(1)

    nutritionist = Nutritionist()

    missing = nutritionist.missing_ingredients(ingredients)
    if missing:
        print(
            "The following ingredients are missing in our database: "
            + ", ".join(missing)
        )
        return

    rating_class = nutritionist.forecast(ingredients)
    print_forecast(rating_class)

    report = nutritionist.nutrition_report(ingredients)
    print_nutrition_report(report)

    top_recipes = nutritionist.similar_recipes(ingredients, top_n=3)
    print_similar_recipes(top_recipes)


if __name__ == "__main__":
    main()
