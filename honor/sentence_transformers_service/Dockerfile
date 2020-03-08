FROM coursera_nlp_honors_serve:latest
LABEL maintainer="Javier Martinez <javiermcebrian@gmail.com>"

RUN pip install flask==1.0.2

RUN mkdir /app
COPY app.py /app
WORKDIR /app

EXPOSE 8080

ENTRYPOINT ["python"]
CMD ["app.py"]