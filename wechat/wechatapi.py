import itchat
import time
import datetime as dt
from apscheduler.schedulers.background import BackgroundScheduler
import random
from itchat.content import *
# import getpost
from getSentimentTree import *

people = []
total_username = []
def get_user(person):
	userName = []
	for element in people:
		users=itchat.search_friends(name =element)
		every_userName= users[0]['UserName']
		userName.append(every_userName)
	print(userName)
	return userName
'''自动回复'''
@itchat.msg_register([TEXT, PICTURE, MAP, CARD, NOTE, SHARING, RECORDING, ATTACHMENT, VIDEO])
def text_reply(msg):
	print(msg['FromUserName'] + ":" + msg['Text'])
	if(msg['Text'] == '情感测试' and (msg['FromUserName'] not in total_username)):
		total_username.append(msg['FromUserName'])
		itchat.send('情感分析（仅支持英文）use TreeLstm with Stanford Sentiment TreeBank',toUserName=msg['FromUserName'])
		itchat.send('回复：T 退出',toUserName=msg['FromUserName'])

	elif msg['FromUserName'] in total_username:
		people_comment = msg['Text']
		if people_comment == "T":
			total_username.remove(msg['FromUserName'])
		else:
			try:
				f = getRootGraph(people_comment)
				itchat.sent_image(f,toUserName = msg['FromUserName'])
			except:
				print("Error!send image fail")
			people_answer = getSentimentTree(people_comment)
			# people_answer = getpost.get_answer(people_comment)
			itchat.send(people_answer, toUserName=msg['FromUserName'])
	
	#people_comment = msg['Text']
	#people_answer = getpost.get_answer(people_comment)
	#itchat.send("自动回复："+people_answer, toUserName=msg['FromUserName'])

if __name__ == '__main__':
	itchat.auto_login() # 默认展示的是图片二维码
	# itchat.auto_login(hotReload=True) # 调试用的，保留登录状态
	total_username = get_user(people)
	for username in total_username:
		itchat.send('你好，我是机器人，很高兴认识您',toUserName=username)
	itchat.run() # 启动微信
