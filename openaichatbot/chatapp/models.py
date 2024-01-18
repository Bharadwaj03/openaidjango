from django.db import models
from django.contrib.auth.models import User


class Chat(models.Model):
    
    message = models.TextField()
    response = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.user.username}: {self.message}'

# Create your models here.
class querydb(models.Model):
    Query = models.CharField(max_length=100)
