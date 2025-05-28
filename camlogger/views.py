from django.shortcuts import render

def index(request):
    return render(request, 'index.html')  # path changed, string in quotes
