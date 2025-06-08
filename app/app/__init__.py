from flask import Flask, render_template, request, redirect, url_for
# from dotenv import load_dotenv
import os

# load_dotenv()


def create_app():
    app = Flask(__name__)
    app.secret_key = "test123"

    from .interface.routes import interface
    app.register_blueprint(interface)



    return app
    