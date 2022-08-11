FROM python:3.9
EXPOSE 8501
WORKDIR /app
COPY requirements_test.txt ./requirements_test.txt
RUN pip install -r requirements_test.txt
COPY . .
CMD streamlit run app.py