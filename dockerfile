# Stage 1: Train models and select the best one
FROM python:3.9 AS model_training

WORKDIR /app

# Copy requirements
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy and run training script
COPY src/model_training.py /app/
RUN python model_training.py

# Stage 2: Serve predictions with best model
FROM python:3.9 AS serving

WORKDIR /app

# Copy trained model artifacts from previous stage
COPY --from=model_training /app/best_model.* /app/
COPY --from=model_training /app/scaler.pkl /app/
COPY --from=model_training /app/model_info.json /app/

# Copy application files
COPY src/main.py /app/
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy templates
COPY src/templates /app/templates
COPY src/statics /app/statics

# Expose port
EXPOSE 4000

# Environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "main.py"]