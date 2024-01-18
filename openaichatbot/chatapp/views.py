from django.shortcuts import render,redirect
from .models import querydb
from .forms import UserInputForm


# import chromadb
# chroma_client= chromadb.Client()
# collection = chroma_client.create_collection(name="my_collection")
# Create your views here.
def index(request):
    return render(request,"Index.html")

def querysubmit(request):
    if request.method=="POST":
        qr= request.POST.get('query')
        obj= querydb(Query=qr)
        obj.save()
        return redirect(index)
    
## User Open AI intraction

# # my_collection = chroma_client.get_collection("my_collection")
# def my_view(request):
#     user_input = ''
#     openai_response = ''
#     all_docs = []
#     if request.method == 'POST':
#         form = UserInputForm(request.POST)
#         if form.is_valid():
#             user_input = form.cleaned_data['user_input']
#             openai_response = get_openai_response(user_input)
#             document_id = uuid.uuid4()
#             my_collection.add(
#                 documents=[user_input],
#                 metadatas=[{"openai_response": openai_response}],
#                 ids=[str(document_id)]
#             )
#         all_docs = my_collection.get()
#         print(all_docs)
#     else:
#         form = UserInputForm()
#         all_docs = my_collection.get()
#     return render(request, 'chatapp.html', {'form': form, 'user_input': user_input, 'openai_response': openai_response, 'all_docs': all_docs})
    
    
from django.http import JsonResponse
import openai


from django.contrib.auth.models import User
from .models import Chat

from django.utils import timezone


openai_api_key = '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'
openai.api_key = openai_api_key

def ask_openai(message):
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an helpful assistant."},
            {"role": "user", "content": message},
        ]
    )
    
    answer = response.choices[0].message.content.strip()
    return answer

# Create your views here.
def chatbot(request):
    

    if request.method == 'POST':
        message = request.POST.get('message')
        response = ask_openai(message)

        # chat = Chat(message=message, response=response, created_at=timezone.now())
        # chat.save()
        return JsonResponse({'message': message, 'response': response})
    return render(request, 'chatbot.html')
