from django.db import models

# Create your models here.


from django.db import models

class File(models.Model):
    name = models.CharField(max_length=255)
    content = models.FileField(upload_to='uploads/')
