from django.contrib import admin

from moon_vision_challenge.classification.models import Inference


class InferenceAdmin(admin.ModelAdmin):
    list_display = ['id', 'status', 'image', 'label']
    list_filter = ['status']
    search_fields = ['id']


admin.site.register(Inference, InferenceAdmin)
