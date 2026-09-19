FROM python:3.12-slim AS base

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# 回归阶段：与交付镜像同一 CPython 与 app 代码，仅追加 pytest 与测试代码。
# 供 compose 的一次性 api-tests 服务使用，跑完即退出，退出码即回归结果。
FROM base AS test

COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY pytest.ini .
COPY tests ./tests

CMD ["pytest"]

FROM base AS runtime

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
