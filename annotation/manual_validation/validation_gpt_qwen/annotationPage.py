from flask import render_template, request, Response, send_file, redirect
from werkzeug.security import safe_join 
import json
import os

# GLOBAL VARIABLE WITH INFO FROM THE APP
CONFIG = {
    "annotation_file": "",
    "port": 0,
    "output_file_not_selected": "",
    "output_file_selected": "",
    "image_folder": "",
    "base_path": "",
    "dataset_dir": "./images",
    "current_index": 0,
    "user": "default",
}

CONFIG["dataset_cache"] = None

def set_config(config):
    CONFIG.update(config)
    
    correct_annotations_file = "correct_annotations_with_proxy.json"
    if os.path.exists(correct_annotations_file):
        try:
            with open(correct_annotations_file, "r") as f:
                CONFIG["correct_cache"] = json.load(f)
        except json.JSONDecodeError:
            CONFIG["correct_cache"] = {}
            print("Warning: Could not load correct annotations cache")
    else:
        CONFIG["correct_cache"] = {}


def get_last_index_file():
    return os.path.join(CONFIG["user"], f"last_index_{CONFIG['user']}.json") 

def load_last_index():
    last_index_file = get_last_index_file()
    if os.path.exists(last_index_file):
        with open(last_index_file, "r") as f:
            data = json.load(f)
            return data.get("current_index", 0)
    return 0

def save_last_index(index):
    last_index_file = get_last_index_file()
    with open(last_index_file, "w") as f:
        json.dump({"current_index": index}, f)

def get_dataset():
    """Loads dataset from annotation file."""
    if CONFIG["dataset_cache"] is None:
        with open(CONFIG["annotation_file"], "r") as f:
            CONFIG["dataset_cache"] = json.load(f)
    return CONFIG["dataset_cache"]

def get_validated_dataset():
    """Loads already validated images, if available."""
    if os.path.exists(CONFIG["output_file_not_selected"]):
        try:
            with open(CONFIG["output_file_not_selected"], "r") as f:
                file_content = f.read().strip()
                if not file_content:
                    return {}
                return json.loads(file_content)
        except json.JSONDecodeError:
            return {}
    return {}

def get_sorted_dataset():
    """Returns dataset sorted by class and year."""

    sorted_dataset = []
    if CONFIG.get("sorted_dataset_cache") is None:
        dataset = get_dataset()
        sorted_dataset = [(cls, period, dataset[cls][period]) for cls in sorted(dataset.keys()) for period in sorted(dataset[cls].keys())]
        CONFIG["sorted_dataset_cache"] = sorted_dataset
    return CONFIG["sorted_dataset_cache"]


def load_saved_selections(cls, period):
    """Loads saved selections from JSON files."""
    selected_file = CONFIG["output_file_selected"]
    not_selected_file = CONFIG["output_file_not_selected"]

    selected_images = []
    
    # Load selected images if file exists
    if os.path.exists(selected_file):
        try:
            with open(selected_file, "r") as f:
                file_content = f.read().strip()
                if not file_content:
                    return {}
                saved_data = json.loads(file_content)
                selected_images = saved_data.get(period, {}).get(cls, [])
        except json.JSONDecodeError:
            return {}
    return selected_images

def path_to_url(path):
    """Converts image path to a URL."""
    parts = path.rsplit("/", 2)
    if len(parts) < 2:
        return Response("Invalid image path format", status=400)
    
    folder, filename = parts[-2], parts[-1]
    return f"/images/{folder}/{filename}"

def sort_images(images):
    # Sort first by model name, then by score (descending)
    return sorted(images, key=(lambda x: (x[4], x[9])), reverse=True)

def get_annotation_form():
    """Handles annotation form submission (POST) and retrieval (GET)."""
    if request.method == "POST":
        data = request.form.get("json_data")
        if not data:
            return Response("Missing 'json_data' field", status=400)

        try:
            json_data = json.loads(data)
            class_name = json_data.get("class_name", "").strip()
            year = json_data.get("year", "").strip()

            if not class_name or not year:
                return Response("Invalid class or year", status=400)

            save_result(
                class_name,
                year,
                json_data.get("unselected_images", []),
                json_data.get("selected_images", [])
            )

            # ✅ Compute new progress after saving
            completed_pairs, total_pairs = get_annotation_progress()
            return json.dumps({"completed_pairs": completed_pairs, "total_pairs": total_pairs})

        except (json.JSONDecodeError, KeyError) as e:
            return Response(f"Invalid JSON format: {str(e)}", status=400)

    # ✅ If GET request, load the annotation page correctly
    dataset = get_sorted_dataset()

    CONFIG["current_index"] = load_last_index()
    
    if 0 <= CONFIG["current_index"] < len(dataset):
        cls, period, images = dataset[CONFIG["current_index"]]
    else:
        # If index is out of range, reset it
        cls, period, images = get_classes_and_period_not_yet_validated()
        if cls is None:
            return Response("No more classes to validate", status=200)

    # ✅ Retrieve dataset and selections
    full_dataset = get_dataset()
    images_with_data = []
    for img_path, img_annotation in images:
        annotation_class = img_annotation["class_name"]
        gpt_pred = img_annotation["pred"]
        model_probability = img_annotation["model_probability"]
        car_probability = img_annotation["car_probability"]
        qwen_class = img_annotation["qwen_class"]
        qwen_pred = img_annotation["qwen_pred"]
        qwen_score = round(img_annotation["qwen_score"]["annotation_score"] * 100, 2)
        top_preds = img_annotation["qwen_score"]["top_preds"]
        top_preds = [[p[0], round(p[1] * 100, 2)] for p in top_preds]
        top_preds = json.dumps(top_preds)
        box = img_annotation["box"]
        img_url = path_to_url(img_path)
        images_with_data.append((img_url, img_path, box, annotation_class, gpt_pred, model_probability, car_probability, qwen_class, qwen_pred, qwen_score, top_preds))

    images_with_data = sort_images(images_with_data)
    selected_images = load_saved_selections(cls, period) #or []
    selected_images_count = len(selected_images)
    total_images_count = len(images_with_data)
    correct_images_count = get_correct_images_count(cls, period)

    completed_pairs, total_pairs = get_annotation_progress()  # Compute progress

    return render_template(
        "annotation.html",
        title=f"{cls} - {period}",
        class_name=cls,
        year=period,
        images=images_with_data,
        selected_images=selected_images,
        selected_images_count=selected_images_count,
        total_images_count=total_images_count,
        correct_images_count=correct_images_count,
        available_classes=sorted(full_dataset.keys()),
        dataset={c: sorted(full_dataset[c].keys()) for c in full_dataset},
        completed_pairs=completed_pairs,
        total_pairs=total_pairs
    )

def save_result_to_file(cls, period, file_path, images):
    """Saves annotation results to the specified file."""
    if CONFIG.get("saved_data_cache") is None:
        CONFIG["saved_data_cache"] = {}
    if file_path not in CONFIG["saved_data_cache"]:
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                CONFIG["saved_data_cache"][file_path] = json.load(f)
        else:
            CONFIG["saved_data_cache"][file_path] = {}
    
    CONFIG["saved_data_cache"][file_path].setdefault(period, {})[cls] = images
    # save only if modified
    with open(file_path, "w") as f:
        json.dump(CONFIG["saved_data_cache"][file_path], f)

def save_result(cls, period, unselected_images, selected_images):
    """Saves both selected and unselected image annotations."""
    save_result_to_file(cls, period, CONFIG["output_file_not_selected"], unselected_images)
    save_result_to_file(cls, period, CONFIG["output_file_selected"], selected_images)

def get_classes_and_period_not_yet_validated():
    """Finds the first class and period that has not yet been validated."""
    dataset = get_dataset()
    validated = get_validated_dataset()
    classes = sorted(dataset.keys())
    for cls in classes:
        periods = dataset[cls]
        sorted_periods = sorted(periods.keys())
        for period in sorted_periods:
            return cls, period, periods[period] # Always return class-time even if it has been validated to allow for re-validation
            """if period not in validated or cls not in validated[period]:
                return cls, period, periods[period]"""
    return None, None, None

def get_image(folder, name):
    """Serves an image from the dataset directory."""
    filename = safe_join(CONFIG["dataset_dir"], folder, name)
    if not os.path.exists(filename):
        return Response(f"Image not found: {filename}", status=404)
    
    return send_file(filename, mimetype="image/gif")





def get_previous_annotation():
    """Loads the previous annotation task, ensuring dataset is included."""
    global CONFIG
    dataset = get_sorted_dataset()
    if CONFIG["current_index"] > 0:
        CONFIG["current_index"] -= 1
    save_last_index(CONFIG["current_index"])
    cls, period, images = dataset[CONFIG["current_index"]]
    images_with_data = []
    for img_path, img_annotation in images:
        annotation_class = img_annotation["class_name"]
        gpt_pred = img_annotation["pred"]
        model_probability = img_annotation["model_probability"]
        car_probability = img_annotation["car_probability"]
        qwen_class = img_annotation["qwen_class"]
        qwen_pred = img_annotation["qwen_pred"]
        qwen_score = round(img_annotation["qwen_score"]["annotation_score"] * 100, 2)
        top_preds = img_annotation["qwen_score"]["top_preds"]
        top_preds = [[p[0], round(p[1] * 100, 2)] for p in top_preds]
        top_preds = json.dumps(top_preds)
        box = img_annotation["box"]
        img_url = path_to_url(img_path)
        images_with_data.append((img_url, img_path, box, annotation_class, gpt_pred, model_probability, car_probability, qwen_class, qwen_pred, qwen_score, top_preds))
        
    images_with_data = sort_images(images_with_data)

    selected_images = load_saved_selections(cls, period) or []
    selected_images_count = len(selected_images)
    total_images_count = len(images_with_data)
    correct_images_count = get_correct_images_count(cls, period)

    # ✅ Pass sorted dataset to template
    full_dataset = get_dataset()
    completed_pairs, total_pairs = get_annotation_progress()
    return render_template(
        "annotation.html",
        title=f"{cls} - {period}",
        class_name=cls,
        year=period,
        images=images_with_data,
        selected_images=selected_images,
        selected_images_count=selected_images_count,
        total_images_count=total_images_count,
        correct_images_count=correct_images_count,
        available_classes=sorted(full_dataset.keys()),
        dataset={c: sorted(full_dataset[c].keys()) for c in full_dataset},
        completed_pairs=completed_pairs,
        total_pairs=total_pairs
    )


def get_next_annotation():
    """Loads the next annotation task, ensuring dataset is included."""
    global CONFIG
    dataset = get_sorted_dataset()

    if CONFIG["current_index"] < len(dataset) - 1:
        CONFIG["current_index"] += 1
    save_last_index(CONFIG["current_index"])

    cls, period, images = dataset[CONFIG["current_index"]]
    images_with_data = []
    for img_path, img_annotation in images:
        annotation_class = img_annotation["class_name"]
        gpt_pred = img_annotation["pred"]
        model_probability = img_annotation["model_probability"]
        car_probability = img_annotation["car_probability"]
        qwen_class = img_annotation["qwen_class"]
        qwen_pred = img_annotation["qwen_pred"]
        qwen_score = round(img_annotation["qwen_score"]["annotation_score"] * 100, 2)
        top_preds = img_annotation["qwen_score"]["top_preds"]
        top_preds = [[p[0], round(p[1] * 100, 2)] for p in top_preds]
        top_preds = json.dumps(top_preds)
        box = img_annotation["box"]
        img_url = path_to_url(img_path)
        images_with_data.append((img_url, img_path, box, annotation_class, gpt_pred, model_probability, car_probability, qwen_class, qwen_pred, qwen_score, top_preds))
    
    images_with_data = sort_images(images_with_data)
    selected_images = load_saved_selections(cls, period) or []
    selected_images_count = len(selected_images)
    total_images_count = len(images_with_data)
    correct_images_count = get_correct_images_count(cls, period)

    # ✅ Pass sorted dataset to template
    full_dataset = get_dataset()
    completed_pairs, total_pairs = get_annotation_progress()
    return render_template(
        "annotation.html",
        title=f"{cls} - {period}",
        class_name=cls,
        year=period,
        images=images_with_data,
        selected_images=selected_images,
        selected_images_count=selected_images_count,
        total_images_count=total_images_count,
        correct_images_count=correct_images_count,
        available_classes=sorted(full_dataset.keys()),
        dataset={c: sorted(full_dataset[c].keys()) for c in full_dataset},
        completed_pairs=completed_pairs,
        total_pairs=total_pairs
    )


def get_specific_annotation():
    """Loads a specific class and year annotation and updates the index for navigation."""
    cls = request.args.get("class")
    period = request.args.get("year")

    if not cls or not period:
        return Response("Missing class or year parameter", status=400)

    dataset = get_sorted_dataset()  # ✅ Use sorted dataset (list of tuples)

    # ✅ Convert period to string for consistency
    period = str(period)

    # ✅ Find the correct index in the sorted dataset
    for i, (c, p, imgs) in enumerate(dataset):  
        if c == cls and str(p) == period:
            CONFIG["current_index"] = i
            save_last_index(i)
            break
    else:
        return Response("Class or year not found in sorted dataset", status=404)

    # ✅ Get images from `get_dataset()` (which is a dictionary)
    full_dataset = get_dataset()
    if cls not in full_dataset or period not in full_dataset[cls]:
        return Response("Class or year not found in dataset", status=404)

    images_with_data = []
    for img_path, img_annotation in full_dataset[cls][period]:
        annotation_class = img_annotation["class_name"]
        gpt_pred = img_annotation["pred"]
        model_probability = img_annotation["model_probability"]
        car_probability = img_annotation["car_probability"]
        qwen_class = img_annotation["qwen_class"]
        qwen_pred = img_annotation["qwen_pred"]
        qwen_score = round(img_annotation["qwen_score"]["annotation_score"] * 100, 2)
        top_preds = img_annotation["qwen_score"]["top_preds"]
        top_preds = [[p[0], round(p[1] * 100, 2)] for p in top_preds]
        top_preds = json.dumps(top_preds)
        box = img_annotation["box"]
        img_url = path_to_url(img_path)
        images_with_data.append((img_url, img_path, box, annotation_class, gpt_pred, model_probability, car_probability, qwen_class, qwen_pred, qwen_score, top_preds))

    images_with_data = sort_images(images_with_data)
    selected_images = load_saved_selections(cls, period) or []
    selected_images_count = len(selected_images)
    total_images_count = len(images_with_data)
    correct_images_count = get_correct_images_count(cls, period)

    completed_pairs, total_pairs = get_annotation_progress()
    return render_template(
        "annotation.html",
        title=f"{cls} - {period}",
        class_name=cls,
        year=period,
        images=images_with_data,
        selected_images=selected_images,
        selected_images_count=selected_images_count,
        total_images_count=total_images_count,
        correct_images_count=correct_images_count,
        available_classes=sorted(full_dataset.keys()),
        dataset={c: sorted(full_dataset[c].keys()) for c in full_dataset},
        completed_pairs=completed_pairs,
        total_pairs=total_pairs
    )



def get_annotation_progress():
    """Calculates progress as the number of annotated class-time pairs."""
    dataset = get_sorted_dataset()
    validated = get_validated_dataset()

    total_pairs = len(dataset)
    completed_pairs = 0

    for cls, period, _ in dataset:
        if period in validated and cls in validated[period]:
            completed_pairs += 1

    return completed_pairs, total_pairs

def get_correct_images_count(cls,period):
    correct_cache = CONFIG.get("correct_cache", {})
    if not correct_cache:
        print("Warning: Correct cache is empty")
        return 0
    return len(correct_cache.get(cls, {}).get(period, []))