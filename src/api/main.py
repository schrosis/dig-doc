import json
import logging

from fastapi import FastAPI, Request

app = FastAPI()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[logging.FileHandler(filename="log.txt")],
)

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def request_handler(request: Request) -> dict[str, str]:
    logging.info(
        json.dumps(
            {
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
                "query": request.url.query,
                "body": str(await request.body()),
            },
        ),
    )
    return {"message": "Hello World?"}
