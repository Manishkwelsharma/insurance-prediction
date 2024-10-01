FROM python:3.11.9
WORKDIR D:\projects\orison_tech\Regression
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8080
CMD ["streamlit", "run", "model.py", "--server.port=8080", "--server.address=0.0.0.0"]