from ultralytics import YOLO
import requests
from dotenv import load_dotenv
import os

def fetch_recipes(ingredient):
    load_dotenv()
    BASE_URL = "https://api.spoonacular.com/recipes"
    SPOONACULAR_API_KEY = os.getenv("SPOONACULAR_API_KEY")
    HEADERS = {"x-api-key": SPOONACULAR_API_KEY}

    url = f"{BASE_URL}/findByIngredients"
    querystring = {"ingredients": ingredient, "number": 3}
    response = requests.get(url, headers=HEADERS, params=querystring)
    
    if response.status_code == 200:
        recipes = response.json()
        if not recipes:
            print(f"No recipes found for ingredient: {ingredient}")
            return

        for recipe in recipes:
            print(f"Recipe: {recipe['title']}")
            print(f"Image: {recipe['image']}")
            
            recipe_id = recipe["id"]
            steps_url = f"{BASE_URL}/{recipe_id}/analyzedInstructions"
            steps_response = requests.get(steps_url, headers=HEADERS)
            
            if steps_response.status_code == 200:
                instructions = steps_response.json()
                steps = []
                if instructions:
                    for instruction in instructions:
                        for step in instruction["steps"]:
                            steps.append(step["step"])
                
                # Print the steps
                print("Steps:")
                for i, step in enumerate(steps, start=1):
                    print(f"\t{i}. {step}")
            else:
                print(f"Error fetching recipe steps for {recipe['title']}")
            print("\n")
    else:
        print(f"Error fetching recipes: {response.status_code}")

def find_items(image_path, model_path="best.pt"):
    model = YOLO(model_path)
    results = model(image_path)
    detected_classes = [model.names[int(box.cls)] for box in results[0].boxes]
    unique_objects = list(set(detected_classes))
    return unique_objects

def main():
    image_path = input("Enter path to image: ")
    items = find_items(image_path, "./runs/detect/train/weights/best.pt")
    for item in items:
        fetch_recipes(item)

if(__name__ == "__main__"):
    main()
