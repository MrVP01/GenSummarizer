import redis

# Initialize Redis connection
cache = redis.StrictRedis(host='redis', port=6379, decode_responses=True)

# Function to store document summaries in Redis cache
def cache_summary(document, summary):
    cache.set(document, summary)

# Function to retrieve cached summary
def get_cached_summary(document):
    return cache.get(document)
