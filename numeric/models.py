from django.db import models

class Data(models.Model):
    s_tabulated = models.FileField(upload_to='uploads/')
    z_tabulated = models.FileField(upload_to='uploads/', )
    density_tabulated = models.FileField(upload_to='uploads/')

