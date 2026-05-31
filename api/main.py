"""FastAPI application entry point for CV-Job Matching API."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from api.routers import ranking, cv, job, evaluation, health

# ---------------------------------------------------------------------------
# Rate limiter — 60 istek/dakika (IP başına); yük testleri için override edilebilir.
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

app = FastAPI(
    title="CV-Job Matching API",
    description="API for matching job postings with candidate CVs",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(job.router, prefix="/api", tags=["Jobs"])
app.include_router(cv.router, prefix="/api", tags=["CVs"])
app.include_router(ranking.router, prefix="/api", tags=["Ranking"])
app.include_router(evaluation.router, prefix="/api", tags=["Evaluation"])


@app.get("/")
async def root():
    return {"message": "CV-Job Matching API", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)