# TODO

## 1. PROJECT SETUP & CLEANUP
- [x] Review README.md for project overview and current progress.
- [x] Remove unnecessary folders (.pytest_cache, old .git) if reinitializing.
- [x] Convert .env.example → .env and add real credentials/configs.
- [x] Verify requirements.txt or pyproject.toml match dependencies used.

## 2. BACKEND DEVELOPMENT
- [x] Set up backend framework (Flask / FastAPI / Django).
- [ ] Define API routes, models, and controllers.
- [ ] Connect database (PostgreSQL / MySQL / MongoDB).
- [ ] Implement authentication & JWT authorization.
- [ ] Add unit tests for endpoints in tests/.


## 3. FRONTEND DEVELOPMENT
- [ ] Inspect or create frontend (React / Vue / Next.js).
- [ ] Build UI layout (Navbar, Dashboard, etc.).
- [ ] Implement API integration with backend.
- [ ] Add routing, state management, and responsive design.
- [ ] Test all frontend functionality (forms, data loading).

## 4. CORE LOGIC / AI FEATURES (src/)
- [ ] Review src/ for main algorithms or logic.
- [ ] Integrate backend APIs with AI / financial logic.
- [ ] Add data validation, error handling, and logging.
- [ ] Ensure model pipelines or analytics modules are connected.

## 5. DOCUMENTATION
- [ ] Update docs/architecture.md with final architecture.
- [ ] Add developer and user documentation under docs/notes and docs/reports.
- [ ] Include setup, testing, and deployment steps in README.md.

## 6. TESTING & DEBUGGING
- [ ] Configure pytest.ini and ensure all tests run successfully.
- [ ] Fix test failures and add new test cases.
- [ ] Add coverage reports and configure CI/CD (GitHub Actions).

## 7. DEPLOYMENT
- [ ] Create Dockerfile and docker-compose.yml.
- [ ] Set up deployment on AWS / Render / Vercel / Railway.
- [ ] Test .env variables in production environment.
- [ ] Verify build and deployment commands.

## 8. FINAL REVIEW
- [ ] Confirm dependencies in requirements.txt or pyproject.toml.
- [ ] Delete unused test files and redundant code.
- [ ] Finalize and polish documentation.

## 9. Create a functionality test report generator
- [ ] Create a script that runs all the tests in the `tests/` directory.
- [ ] The script will generate a report that shows the status of each test (pass/fail).
- [ ] The report will be in HTML format.
- [ ] The report will be saved in the `reports/` directory.
- [ ] Install any new dependencies required for the test report generator.
- [ ] Run the test report generator script to generate the report.
- [ ] Verify that the report is generated correctly.
- [ ] Add the test report generator script to the CI/CD pipeline.
