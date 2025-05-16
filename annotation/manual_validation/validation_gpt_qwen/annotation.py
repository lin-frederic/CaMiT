from flask import Flask, redirect
import argparse
import annotationPage
import os
import json

def redirect_to_annotation():
    return redirect("/get_annotation_form", code=302)

def get_app() -> Flask:
    app = Flask(__name__)
    app.add_url_rule("/", view_func=redirect_to_annotation)
    app.add_url_rule("/get_annotation_form", "annotation", view_func=annotationPage.get_annotation_form, methods=["GET", "POST"])
    app.add_url_rule("/images/<folder>/<name>", "images", view_func=annotationPage.get_image)
    app.add_url_rule("/get_previous_annotation", "previous", view_func=annotationPage.get_previous_annotation)
    app.add_url_rule("/get_next_annotation", "next", view_func=annotationPage.get_next_annotation)
    app.add_url_rule("/get_specific_annotation", "specific", view_func=annotationPage.get_specific_annotation)
    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Test set validation")
    parser.add_argument("annotation_file", type=str)
    parser.add_argument("port", type=int)
    #parser.add_argument("--output_file_not_selected", type=str, default="not_selected.json")
    #parser.add_argument("--output_file_selected", type=str, default="selected.json")
    parser.add_argument("--user", type=str, default="default")
    parser.add_argument("--dataset_dir", type=str, default="", help="will default to images folder in the application")
    args = parser.parse_args()

    os.makedirs(args.user, exist_ok=True)
    args.output_file_selected = os.path.join(args.user, f"selected_{args.user}.json")
    args.output_file_not_selected = os.path.join(args.user, f"not_selected_{args.user}.json")
    args.last_index_file = os.path.join(args.user, f"last_index_{args.user}.json")

    if args.dataset_dir == "":
        args.dataset_dir = os.path.join(os.path.dirname(__file__), "images")
        
    if not os.path.exists(args.last_index_file):
        with open(args.last_index_file, "w") as f:
            json.dump({"index": 0}, f)
                
    annotationPage.set_config(vars(args))
    app = get_app()
    app.run(port=args.port)




