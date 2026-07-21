# Barrikade API - Postman Collection

Postman collection covering all Barrikade Detection API endpoints
(health, stateless detect, and the full session lifecycle).

## Setup
1. Import `barrikade-core.postman_collection.json` into Postman.
2. Use the **Postman desktop app** - the web client cannot reach `localhost`.
3. Set the `baseUrl` collection variable to your API address
   (`http://localhost:8000` for local Docker).

## Running the session flow
Run **Create Session** first. It saves the returned `session_id` into a
collection variable via a post-response script, so the other session
requests (`detect`, `get`, `end`, `report`) auto-fill it. Run them in order:
create → detect → get → end → report.

## Start the API
From the repo root: `docker compose up --build`
Confirm readiness at `GET /health/ready` before sending requests.