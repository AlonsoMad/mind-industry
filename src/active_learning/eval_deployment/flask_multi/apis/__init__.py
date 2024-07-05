"""
from flask_restx import Api

from .namespace import api as n1

api = Api(
    title="EWB's Topic Modeling API",
    version='1.0',
    description='whatever',
)

api.add_namespace(n1, path='/test')
"""