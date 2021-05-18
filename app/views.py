from django.shortcuts import render, HttpResponse, redirect
from .bert.preprocess import create_inputs_targets, create_squad_examples, get_model_inference

# Create your views here.

def index(request):
    return redirect("predict/")

def predict(request):
    if request.method == "POST":
        question = request.POST["question"]
        #print(question)
        context = request.POST["context"]
        #print(context)

        data = {"data":
        [
        {"title": "Project Apollo",
         "paragraphs": [
             {
                 "context": context,
                 "qas": [
                     {"question": question,
                      "id": "Q1"
                      },
                 ]}]}]}
        print(data)

        test_samples = create_squad_examples(data) # preproc
        x_test, _ = create_inputs_targets(test_samples) # preproc
        pred_answers = get_model_inference(x_test, test_samples)

        return render(request, "app/predict.html", {"question": question, "answer": pred_answers})

    return render(request, "app/collectResponse.html", {})