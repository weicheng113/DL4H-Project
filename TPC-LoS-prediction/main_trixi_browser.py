from trixi.experiment_browser.browser import create_flask_app, register_url_routes
import os
import argparse
from flask import Blueprint, Flask


def run_trixi_browser():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_directory",
                        help="Give the path to the base directory of your project files",
                        type=str, default=".")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Turn debug mode on, eg. for live reloading.")
    parser.add_argument("-x", "--expose", action="store_true",
                        help="Make server externally visible")
    parser.add_argument("-p", "--port", default=5000, type=int,
                        help="Port to start the server on (5000 by default)")
    args = [
       # "./experiment_results/experiment_20220416/final/eICU/LoS/TPC",
        "./notebooks/experiment_dir",
        "--port", "5000",
        "--debug",
        "--expose"
    ]
    args = parser.parse_args(args=args)

    base_dir = args.base_directory

    base_dir = os.path.abspath(base_dir)

    app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), "static"))
    blueprint = Blueprint("data", __name__, static_url_path='/' + base_dir, static_folder=base_dir)
    app.register_blueprint(blueprint)

    register_url_routes(app, base_dir)

    host = "0.0.0.0" if args.expose else "localhost"
    app.run(debug=args.debug, host=host, port=args.port)


if __name__ == '__main__':
    run_trixi_browser()
