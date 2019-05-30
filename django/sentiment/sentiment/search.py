from django.shortcuts import render
from django.views.decorators import csrf
from . import getContent_lstm
# from . import getContent_lstm_glove
# from . import getContent_textcnn
# from . import getContent_textcnn_glove
# from . import getContent_imdb_lstm
# from . import getContent_imdb_cnn
# from . import getContent_Renn
# from . import getContent_treelstm
def search_post(request):
    ctx ={}
    if request.POST:
        # ctx['rlt'] = request.POST['q']
        try:
            ctx['rlt1'] = getContent_lstm.toStr(request.POST['q'])
        except:
            pass

        try:
            ctx['rlt2'] = getContent_lstm_glove.toStr(request.POST['q'])
        except:
            pass

        try:
            ctx['rlt3'] = getContent_textcnn.toStr(request.POST['q'])
        except:
            pass

        try:
            ctx['rlt4'] = getContent_textcnn_glove.toStr(request.POST['q'])
        except:
            pass

        try:
            ctx['rlt5'] = getContent_imdb_lstm.toStr(request.POST['q'])
        except:
            pass

        try:
            ctx['rlt6'] = getContent_imdb_cnn.toStr(request.POST['q'])
        except:
            pass
        try:
            ctx['rlt7'] = getContent_Renn.toStr(request.POST['q'])
        except:
            pass

        try:
            ctx['rlt8'] = getContent_treelstm.toStr(request.POST['q'])
        except:
            pass
        
    return render(request, "post.html", ctx)

if __name__ == '__main__':
    pass
    # try:
    #   print(getContent_lstm.toStr("I love you."))
    # except:
    #   pass
    # try:
    #   print(getContent_lstm.toStr("I love you."))
    # except:
    #   pass
    # try:
    #   print(getContent_textcnn.toStr("I love you."))
    # except:
    #   pass
    # try:
    #   print(getContent_textcnn_glove.toStr("I love you."))
    # except:
    #   pass