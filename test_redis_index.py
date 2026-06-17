import redis
import os
from redis.commands.search.field import NumericField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType

client = redis.Redis.from_url('redis://default:Iz7EkhAc5rG4qSOpzDMs2CU5mvn9WWiO@calculator-wisplike-macrobright-75019.db.redis.io:11899')

schema = [
    TextField("query"),
    TextField("response"),
    NumericField("created_at"),
    VectorField(
        "vector",
        "FLAT",
        {
            "TYPE": "FLOAT32",
            "DIM": 512,
            "DISTANCE_METRIC": "COSINE",
        },
    ),
]
try:
    client.ft("idx:faq_cache").create_index(
        schema,
        definition=IndexDefinition(
            prefix=["faq_resp:sem:"], index_type=IndexType.HASH
        ),
    )
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
