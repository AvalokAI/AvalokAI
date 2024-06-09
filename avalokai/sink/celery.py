from celery import Celery

app = Celery("proj", broker="amqp://guest@localhost//", include=["sink.tasks"])

# Optional configuration, see the application user guide.
app.conf.update(
    result_expires=3600,
    event_serializer="pickle",
    task_serializer="pickle",
    result_serializer="pickle",
    accept_content=["pickle"],
)

if __name__ == "__main__":
    app.start()
