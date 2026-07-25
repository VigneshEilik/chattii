from fastapi import FastAPI

app = FastAPI(
    title="Chatii API",
    description="OpenAI-compatible HTTP serving engine for the Chatii model framework.",
    version="0.1.0-dev",
)


@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "chatii-10m"}


@app.get("/")
def index():
    return {
        "message": (
            "Welcome to Chatii API Server. See /docs for Swagger UI documentation."
        )
    }
